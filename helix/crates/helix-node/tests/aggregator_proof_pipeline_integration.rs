//! Aggregator Proof Pipeline Integration Tests.
//!
//! Tests the full ZK proof generation → on-chain submission pipeline using Anvil.
//! Each test deploys fresh contracts, creates an `AggregatorProofPipeline`,
//! and verifies proofs are accepted by the `HelixCoordinatorV2` contract.
//!
//! # Requirements
//!
//! - Foundry (`forge`, `anvil`) must be installed and in PATH
//! - Contracts must compile via `forge build`
//!
//! # Note
//!
//! These tests involve real ZK proof generation (k=14 Halo2 circuits) and
//! are slow (~6-10s per proof in debug mode). Run with `--release` for faster execution.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ethers::abi::{Abi, Token};
use ethers::contract::{Contract, ContractFactory};
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};
use ethers::utils::Anvil;

use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

use helix_node::aggregator_proof_pipeline::{
    AggregatorModelState, AggregatorProofPipeline, ProofPipelineConfig,
};
use helix_node::sc_client::SCClient;

// ============================================================================
// Test Infrastructure
// ============================================================================

/// Returns the path to the contracts directory (relative to helix-node crate).
fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("helix dir")
        .join("contracts")
}

/// Ensures contracts are compiled and returns the Foundry output directory.
fn ensure_contracts_compiled() -> PathBuf {
    let dir = contracts_dir();
    let out = dir.join("out");

    let artifact = out
        .join("HelixCoordinatorV2.sol")
        .join("HelixCoordinatorV2.json");

    if artifact.exists() {
        return out;
    }

    eprintln!("Compiling contracts with forge build...");
    let status = std::process::Command::new("forge")
        .arg("build")
        .current_dir(&dir)
        .status()
        .expect("forge build failed — is Foundry installed?");
    assert!(status.success(), "forge build failed");
    out
}

/// Loads a contract ABI + bytecode from a Foundry compiled artifact.
fn load_contract_artifact(out_dir: &Path, sol_file: &str, contract_name: &str) -> (Abi, Bytes) {
    let path = out_dir
        .join(sol_file)
        .join(format!("{contract_name}.json"));

    let json: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open artifact {path:?}: {e}")),
    )
    .expect("Failed to parse artifact JSON");

    let abi: Abi = serde_json::from_value(json["abi"].clone()).expect("Failed to parse ABI");

    let bytecode_hex = json["bytecode"]["object"]
        .as_str()
        .expect("No bytecode in artifact");
    let hex_str = bytecode_hex.strip_prefix("0x").unwrap_or(bytecode_hex);
    let bytecode = Bytes::from(hex::decode(hex_str).expect("Invalid bytecode hex"));

    (abi, bytecode)
}

/// Computes `keccak256(abi.encodePacked(lo, hi))` matching the contract's `_hashPair()`.
fn compute_hash_pair(lo: U256, hi: U256) -> U256 {
    let mut data = [0u8; 64];
    lo.to_big_endian(&mut data[0..32]);
    hi.to_big_endian(&mut data[32..64]);
    let hash = ethers::utils::keccak256(&data);
    U256::from_big_endian(&hash)
}

/// Converts Fr to U256 (little-endian repr).
fn fr_to_u256(fr: &Fr) -> U256 {
    let repr = fr.to_repr();
    U256::from_little_endian(repr.as_ref())
}

type SignedClient = Arc<SignerMiddleware<Provider<Http>, LocalWallet>>;

/// Anvil-backed test environment with deployed MockVerifier + HelixCoordinatorV2.
struct AnvilTestEnv {
    /// Anvil instance (killed on drop).
    #[allow(dead_code)]
    anvil: ethers::utils::AnvilInstance,
    /// Signed HTTP provider for direct contract calls.
    client: SignedClient,
    /// SCClient for the AggregatorProofPipeline.
    sc_client: Arc<SCClient>,
    /// Deployed MockVerifier contract (for setAccept).
    mock_verifier: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
    /// Deployed HelixCoordinatorV2 address.
    coordinator_addr: Address,
    /// Deployer/treasury address.
    #[allow(dead_code)]
    deployer: Address,
    /// The contract's maxErrorBound.
    #[allow(dead_code)]
    max_error_bound: U256,
}

impl AnvilTestEnv {
    /// Deploys MockVerifier + HelixCoordinatorV2 on a fresh Anvil instance.
    async fn new() -> Self {
        let out_dir = ensure_contracts_compiled();

        // Spawn Anvil with increased code size limit (HelixCoordinatorV2 exceeds default 24KB)
        let anvil = Anvil::new()
            .arg("--code-size-limit")
            .arg("100000")
            .spawn();
        let provider =
            Provider::<Http>::try_from(anvil.endpoint()).expect("Failed to connect to Anvil");

        // Create wallet from first Anvil account
        let wallet: LocalWallet = anvil.keys()[0].clone().into();
        let wallet = wallet.with_chain_id(anvil.chain_id());
        let deployer = wallet.address();
        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        // Deploy MockVerifierForDeploy
        let (mock_abi, mock_bytecode) =
            load_contract_artifact(&out_dir, "Deploy.s.sol", "MockVerifierForDeploy");
        let mock_factory = ContractFactory::new(mock_abi.clone(), mock_bytecode, client.clone());
        let mock_contract = mock_factory
            .deploy(())
            .expect("MockVerifier deploy args")
            .send()
            .await
            .expect("MockVerifier deploy failed");
        let mock_verifier_addr = mock_contract.address();

        // Deploy HelixCoordinatorV2(verifier, treasury)
        let (coord_abi, coord_bytecode) =
            load_contract_artifact(&out_dir, "HelixCoordinatorV2.sol", "HelixCoordinatorV2");
        let coord_factory =
            ContractFactory::new(coord_abi.clone(), coord_bytecode, client.clone());
        let coord_contract = coord_factory
            .deploy((
                Token::Address(mock_verifier_addr),
                Token::Address(deployer),
            ))
            .expect("Coordinator deploy args")
            .send()
            .await
            .expect("Coordinator deploy failed");
        let coordinator_addr = coord_contract.address();

        // Read maxErrorBound
        let coordinator = Contract::new(coordinator_addr, coord_abi, client.clone());
        let max_error_bound: U256 = coordinator
            .method::<_, U256>("maxErrorBound", ())
            .expect("maxErrorBound method")
            .call()
            .await
            .expect("maxErrorBound call failed");

        // Create SCClient from Anvil's first account
        let private_key_hex = hex::encode(anvil.keys()[0].to_bytes());
        let sc_client = SCClient::with_config(
            &anvil.endpoint(),
            &private_key_hex,
            &format!("{coordinator_addr:?}"),
        )
        .await
        .expect("SCClient creation failed");

        let mock_verifier = Contract::new(mock_verifier_addr, mock_abi, client.clone());

        Self {
            anvil,
            client,
            sc_client: Arc::new(sc_client),
            mock_verifier,
            coordinator_addr,
            deployer,
            max_error_bound,
        }
    }

    /// Sets the MockVerifier to accept or reject all proofs.
    async fn set_mock_accept(&self, accept: bool) {
        let _receipt: ethers::types::TransactionReceipt = self
            .mock_verifier
            .method::<_, ()>("setAccept", accept)
            .expect("setAccept method")
            .send()
            .await
            .expect("setAccept send failed")
            .await
            .expect("setAccept confirm failed")
            .expect("setAccept receipt missing");
    }

    /// Reads the model's currentCommitment from the contract via `getModelState`.
    async fn model_commitment(&self, model_id: u64) -> U256 {
        let state = self
            .sc_client
            .get_model_state(model_id)
            .await
            .expect("get_model_state failed");
        state.current_commitment
    }

    /// Reads the last step number for a model.
    async fn last_step_number(&self, model_id: u64) -> U256 {
        let coordinator = Contract::new(
            self.coordinator_addr,
            serde_json::from_str::<Abi>(
                r#"[{"inputs":[{"name":"","type":"uint256"}],"name":"lastStepNumber","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]"#,
            )
            .unwrap(),
            self.client.clone(),
        );
        coordinator
            .method::<_, U256>("lastStepNumber", U256::from(model_id))
            .expect("lastStepNumber method")
            .call()
            .await
            .expect("lastStepNumber call failed")
    }

    /// Reads accumulated error bound for a model.
    async fn accumulated_error(&self, model_id: u64) -> U256 {
        let coordinator = Contract::new(
            self.coordinator_addr,
            serde_json::from_str::<Abi>(
                r#"[{"inputs":[{"name":"","type":"uint256"}],"name":"accumulatedErrorBound","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]"#,
            )
            .unwrap(),
            self.client.clone(),
        );
        coordinator
            .method::<_, U256>("accumulatedErrorBound", U256::from(model_id))
            .expect("accumulatedErrorBound method")
            .call()
            .await
            .expect("accumulatedErrorBound call failed")
    }
}

/// Creates a small 2×2×1 model state with deterministic initial weights.
fn test_model_state() -> AggregatorModelState {
    // Small model: 2 inputs → 2 hidden → 1 output
    // w1: 2×2 = 4 elements, b1: 2 elements, w2: 1×2 = 2 elements, b2: 1 element
    let w1 = vec![
        Fr::from(1u64),
        Fr::from(2u64),
        Fr::from(3u64),
        Fr::from(1u64),
    ];
    let b1 = vec![Fr::from(0u64), Fr::from(0u64)];
    let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
    let b2 = vec![Fr::from(0u64)];

    AggregatorModelState::new(w1, b1, w2, b2)
}

/// Creates a pipeline config that matches the contract's defaults.
fn test_pipeline_config(model_id: u64) -> ProofPipelineConfig {
    let mut config = ProofPipelineConfig::test_small();
    config.model_id = model_id;
    config
}

// ============================================================================
// Tests
// ============================================================================

/// Single proof: register model → stake → start round → prove_and_submit → verify on-chain.
#[tokio::test]
async fn test_single_proof_submission() {
    let env = AnvilTestEnv::new().await;

    let state = test_model_state();
    let initial_lo = fr_to_u256(&state.commitment.0);
    let initial_hi = fr_to_u256(&state.commitment.1);
    let initial_hash = compute_hash_pair(initial_lo, initial_hi);

    // Register model via SCClient directly (to get the model ID)
    let (_receipt, model_id) = env
        .sc_client
        .register_model("helix-test-model", initial_hash, U256::from(0u64), (2, 2, 1, 2, 0))
        .await
        .expect("register_model failed");

    // Stake 1 ETH
    env.sc_client
        .stake(model_id, U256::from(1_000_000_000_000_000_000u64))
        .await
        .expect("stake failed");

    // Start round with 1-hour deadline
    env.sc_client
        .start_round(model_id, 3600)
        .await
        .expect("start_round failed");

    // Verify on-chain commitment matches our initial state
    let on_chain_commitment = env.model_commitment(model_id).await;
    assert_eq!(on_chain_commitment, initial_hash, "Initial commitment mismatch");

    // Create pipeline with the registered model ID
    let config = test_pipeline_config(model_id);
    let pipeline = AggregatorProofPipeline::new(env.sc_client.clone(), state, config);

    // Generate proof and submit on-chain
    let result = pipeline
        .prove_and_submit(1, None, None)
        .await
        .expect("prove_and_submit failed");

    // Verify result fields
    assert_eq!(result.step_number, 1);
    assert_eq!(result.round_id, 1);
    assert!(result.proof_size > 0, "Proof should have non-zero size");
    assert_ne!(
        result.old_commitment, result.new_commitment,
        "Training step should change commitment"
    );
    assert!(result.gas_used > 0, "Should consume gas");

    // Verify on-chain state was updated
    let new_on_chain_commitment = env.model_commitment(model_id).await;
    let expected_new_hash = compute_hash_pair(
        fr_to_u256(&result.new_commitment.0),
        fr_to_u256(&result.new_commitment.1),
    );
    assert_eq!(
        new_on_chain_commitment, expected_new_hash,
        "On-chain commitment should match proof's new_hash"
    );

    // Step number should have advanced
    let step = env.last_step_number(model_id).await;
    assert_eq!(step, U256::from(1u64));

    // Error should have accumulated
    let error = env.accumulated_error(model_id).await;
    assert!(error > U256::zero(), "Error should accumulate after proof");

    // Pipeline stats
    let (submitted, rejected) = pipeline.stats().await;
    assert_eq!(submitted, 1);
    assert_eq!(rejected, 0);
}

/// Three sequential proofs in the same round, verifying commitment chaining.
///
/// Each proof's `new_hash` becomes the next proof's `old_hash` on-chain.
/// The pipeline maintains this chain internally via `AggregatorModelState`.
#[tokio::test]
async fn test_commitment_chaining_3_steps() {
    let env = AnvilTestEnv::new().await;

    let state = test_model_state();
    let initial_lo = fr_to_u256(&state.commitment.0);
    let initial_hi = fr_to_u256(&state.commitment.1);
    let initial_hash = compute_hash_pair(initial_lo, initial_hi);

    // Setup: register, stake, start round
    let (_receipt, model_id) = env
        .sc_client
        .register_model("helix-chain-test", initial_hash, U256::from(0u64), (2, 2, 1, 2, 0))
        .await
        .expect("register_model failed");

    env.sc_client
        .stake(model_id, U256::from(1_000_000_000_000_000_000u64))
        .await
        .expect("stake failed");

    env.sc_client
        .start_round(model_id, 3600)
        .await
        .expect("start_round failed");

    let config = test_pipeline_config(model_id);
    let pipeline = AggregatorProofPipeline::new(env.sc_client.clone(), state, config);

    let mut commitments = vec![initial_hash];

    // Submit 3 sequential proofs
    for step in 1..=3u64 {
        let result = pipeline
            .prove_and_submit(1, None, None)
            .await
            .unwrap_or_else(|e| panic!("prove_and_submit step {step} failed: {e}"));

        assert_eq!(result.step_number, step);

        let new_hash = compute_hash_pair(
            fr_to_u256(&result.new_commitment.0),
            fr_to_u256(&result.new_commitment.1),
        );

        // The old commitment should match what we recorded from the previous step
        let old_hash = compute_hash_pair(
            fr_to_u256(&result.old_commitment.0),
            fr_to_u256(&result.old_commitment.1),
        );
        assert_eq!(
            old_hash,
            *commitments.last().unwrap(),
            "Step {step}: old_hash should chain from previous step"
        );

        commitments.push(new_hash);

        // Verify on-chain commitment was updated
        let on_chain = env.model_commitment(model_id).await;
        assert_eq!(
            on_chain, new_hash,
            "Step {step}: on-chain commitment should match new_hash"
        );
    }

    // Verify final on-chain state
    let final_step = env.last_step_number(model_id).await;
    assert_eq!(final_step, U256::from(3u64));

    // Each consecutive pair should differ (training changes weights each step).
    // Note: with the same training data and aggressive learning rate, some steps
    // might cycle back to a previous state, so we only check consecutive pairs.
    for i in 0..commitments.len() - 1 {
        assert_ne!(
            commitments[i],
            commitments[i + 1],
            "Consecutive commitments at step {i} and {} should differ",
            i + 1
        );
    }

    // Pipeline stats
    let (submitted, rejected) = pipeline.stats().await;
    assert_eq!(submitted, 3);
    assert_eq!(rejected, 0);

    // Pipeline's internal state should reflect step 3
    assert_eq!(pipeline.current_step().await, 3);
}

/// MockVerifier set to reject → proof is rejected with `ProofRejected` error.
#[tokio::test]
async fn test_rejected_proof_handling() {
    let env = AnvilTestEnv::new().await;

    let state = test_model_state();
    let initial_lo = fr_to_u256(&state.commitment.0);
    let initial_hi = fr_to_u256(&state.commitment.1);
    let initial_hash = compute_hash_pair(initial_lo, initial_hi);

    let (_receipt, model_id) = env
        .sc_client
        .register_model("helix-reject-test", initial_hash, U256::from(0u64), (2, 2, 1, 2, 0))
        .await
        .expect("register_model failed");

    env.sc_client
        .stake(model_id, U256::from(1_000_000_000_000_000_000u64))
        .await
        .expect("stake failed");

    env.sc_client
        .start_round(model_id, 3600)
        .await
        .expect("start_round failed");

    // Set MockVerifier to REJECT all proofs
    env.set_mock_accept(false).await;

    let config = test_pipeline_config(model_id);
    let pipeline = AggregatorProofPipeline::new(env.sc_client.clone(), state, config);

    // prove_and_submit should fail because the verifier rejects the proof
    let result = pipeline.prove_and_submit(1, None, None).await;
    assert!(
        result.is_err(),
        "Should fail when verifier rejects proof"
    );

    // The on-chain commitment should NOT have changed (proof was rejected)
    let on_chain = env.model_commitment(model_id).await;
    assert_eq!(
        on_chain, initial_hash,
        "Commitment should not change after rejected proof"
    );

    // Step number should still be 0
    let step = env.last_step_number(model_id).await;
    assert_eq!(step, U256::from(0u64));

    // Pipeline's internal state should not have advanced
    assert_eq!(pipeline.current_step().await, 0);
}

/// Verifies gas consumption is within expected bounds.
#[tokio::test]
async fn test_gas_consumption_bounds() {
    let env = AnvilTestEnv::new().await;

    let state = test_model_state();
    let initial_lo = fr_to_u256(&state.commitment.0);
    let initial_hi = fr_to_u256(&state.commitment.1);
    let initial_hash = compute_hash_pair(initial_lo, initial_hi);

    let (_receipt, model_id) = env
        .sc_client
        .register_model("helix-gas-test", initial_hash, U256::from(0u64), (2, 2, 1, 2, 0))
        .await
        .expect("register_model failed");

    env.sc_client
        .stake(model_id, U256::from(1_000_000_000_000_000_000u64))
        .await
        .expect("stake failed");

    env.sc_client
        .start_round(model_id, 3600)
        .await
        .expect("start_round failed");

    let config = test_pipeline_config(model_id);
    let pipeline = AggregatorProofPipeline::new(env.sc_client.clone(), state, config);

    let result = pipeline
        .prove_and_submit(1, None, None)
        .await
        .expect("prove_and_submit failed");

    // Gas includes Poseidon error checksum verification (65 rounds) which is gas-heavy.
    // With MockVerifier: ~7M gas. With real Halo2Verifier: ~7.5M gas.
    assert!(
        result.gas_used > 100_000,
        "Should use at least 100k gas (got {})",
        result.gas_used
    );
    assert!(
        result.gas_used < 15_000_000,
        "Should use less than 15M gas (got {})",
        result.gas_used
    );

    // Proof size should be 1856 bytes (PSE Keccak256 transcript format)
    assert_eq!(
        result.proof_size, 1856,
        "Proof should be 1856 bytes (PSE format)"
    );
}
