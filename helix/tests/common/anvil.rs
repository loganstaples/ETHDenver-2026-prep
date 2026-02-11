//! Anvil test environment and contract deployment helpers.
//!
//! Provides utilities for spinning up Anvil, deploying HELIX contracts,
//! and interacting with the on-chain infrastructure from Rust integration tests.
//!
//! # Requirements
//!
//! - Foundry (`forge`, `anvil`) must be installed and in PATH
//! - Contracts must be compilable via `forge build`
//!
//! # Usage
//!
//! ```rust,ignore
//! let env = OnChainTestEnv::new().await;
//! // env.coordinator, env.mock_verifier are deployed contracts
//! // env.client() is a signer middleware for transactions
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ethers::abi::{Abi, Token};
use ethers::contract::{Contract, ContractFactory};
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, TransactionReceipt, U256};
use ethers::utils::Anvil;

use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

// ============================================================================
// Type Aliases
// ============================================================================

/// Signed HTTP client type used throughout on-chain tests.
pub type SignedClient = Arc<SignerMiddleware<Provider<Http>, LocalWallet>>;

// ============================================================================
// Path Helpers
// ============================================================================

/// Returns the path to the contracts directory.
pub fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tests dir has parent")
        .join("contracts")
}

/// Ensures contracts are compiled and returns the Foundry output directory.
///
/// Runs `forge build` if the compiled artifacts are not found.
pub fn ensure_contracts_compiled() -> PathBuf {
    let dir = contracts_dir();
    let out = dir.join("out");

    // Check if already compiled
    let coord_artifact = out
        .join("HelixCoordinatorV2.sol")
        .join("HelixCoordinatorV2.json");

    if coord_artifact.exists() {
        return out;
    }

    eprintln!("Compiling contracts with forge build...");
    let status = std::process::Command::new("forge")
        .arg("build")
        .current_dir(&dir)
        .status()
        .expect("Failed to run forge build — is Foundry installed?");

    assert!(status.success(), "forge build failed");
    out
}

/// Loads a contract's ABI and bytecode from a Foundry compiled artifact.
pub fn load_contract_artifact(
    out_dir: &Path,
    sol_file: &str,
    contract_name: &str,
) -> (Abi, Bytes) {
    let path = out_dir
        .join(sol_file)
        .join(format!("{}.json", contract_name));

    let json: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open artifact {:?}: {}", path, e)),
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

// ============================================================================
// Field Element Conversion
// ============================================================================

/// Converts a BN254 Fr element to ethers U256 (big-endian interpretation).
pub fn fr_to_u256(fr: &Fr) -> U256 {
    let repr = fr.to_repr();
    // Fr::to_repr() is little-endian; reverse for big-endian U256
    let mut be = [0u8; 32];
    for (i, b) in repr.as_ref().iter().enumerate() {
        be[31 - i] = *b;
    }
    U256::from_big_endian(&be)
}

/// Converts ethers U256 to BN254 Fr element.
pub fn u256_to_fr(val: U256) -> Fr {
    let mut be = [0u8; 32];
    val.to_big_endian(&mut be);
    // Reverse to little-endian for Fr::from_repr
    let mut le = [0u8; 32];
    for (i, b) in be.iter().enumerate() {
        le[31 - i] = *b;
    }
    Fr::from_repr(le.into()).expect("U256 out of Fr range")
}

// ============================================================================
// Solidity-Compatible Hash Functions
// ============================================================================

/// Computes keccak256(abi.encodePacked(lo, hi)) matching Solidity's `_hashPair`.
///
/// Both `lo` and `hi` are encoded as 32-byte big-endian values (total 64 bytes).
pub fn compute_hash_pair(lo: U256, hi: U256) -> U256 {
    let mut data = [0u8; 64];
    lo.to_big_endian(&mut data[0..32]);
    hi.to_big_endian(&mut data[32..64]);
    let hash = ethers::utils::keccak256(&data);
    U256::from_big_endian(&hash)
}


// ============================================================================
// Test Environment
// ============================================================================

/// Complete on-chain test environment with Anvil + deployed contracts.
///
/// Automatically compiles contracts, spawns Anvil, deploys MockVerifier
/// and HelixCoordinatorV2, and provides a signed client for transactions.
pub struct OnChainTestEnv {
    /// Anvil instance (killed on drop).
    #[allow(dead_code)]
    pub anvil: ethers::utils::AnvilInstance,
    /// Signed HTTP provider (first Anvil account).
    client: SignedClient,
    /// Deployed MockVerifier address.
    pub mock_verifier_addr: Address,
    /// Deployed HelixCoordinatorV2 address.
    pub coordinator_addr: Address,
    /// MockVerifier contract instance (for setAccept etc).
    pub mock_verifier: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
    /// HelixCoordinatorV2 contract instance.
    pub coordinator: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
    /// Deployer/treasury address.
    pub deployer: Address,
    /// The max error bound configured in the coordinator.
    pub max_error_bound: U256,
}

impl OnChainTestEnv {
    /// Creates a new test environment: compiles contracts, spawns Anvil, deploys contracts.
    pub async fn new() -> Self {
        let out_dir = ensure_contracts_compiled();

        // Spawn Anvil
        let anvil = Anvil::new().spawn();
        let provider = Provider::<Http>::try_from(anvil.endpoint())
            .expect("Failed to connect to Anvil");

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
        let coord_factory = ContractFactory::new(coord_abi.clone(), coord_bytecode, client.clone());
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

        // Create typed contract instances
        let mock_verifier = Contract::new(mock_verifier_addr, mock_abi, client.clone());
        let coordinator = Contract::new(coordinator_addr, coord_abi, client.clone());

        // Read max error bound from contract
        let max_error_bound: U256 = coordinator
            .method::<_, U256>("maxErrorBound", ())
            .expect("maxErrorBound method")
            .call()
            .await
            .expect("maxErrorBound call failed");

        Self {
            anvil,
            client,
            mock_verifier_addr,
            coordinator_addr,
            mock_verifier,
            coordinator,
            deployer,
            max_error_bound,
        }
    }

    /// Returns the signed client.
    pub fn client(&self) -> SignedClient {
        self.client.clone()
    }

    /// Registers a model with the given initial commitment.
    /// Returns the model ID.
    pub async fn register_model(&self, initial_commitment: U256) -> U256 {
        // Read nextModelId before calling
        let next_id: u32 = self
            .coordinator
            .method::<_, u32>("nextModelId", ())
            .expect("nextModelId method")
            .call()
            .await
            .expect("nextModelId call failed");
        let model_id = U256::from(next_id);

        // Register model
        let _receipt: TransactionReceipt = self
            .coordinator
            .method::<_, ()>(
                "registerModel",
                (
                    "ipfs://helix-test-model".to_string(),
                    initial_commitment,
                    U256::zero(), // use default minStake
                ),
            )
            .expect("registerModel method")
            .send()
            .await
            .expect("registerModel send failed")
            .await
            .expect("registerModel confirm failed")
            .expect("registerModel receipt missing");

        model_id
    }

    /// Stakes ETH for a model.
    pub async fn stake(&self, model_id: U256, amount_wei: U256) {
        let _receipt: TransactionReceipt = self
            .coordinator
            .method::<_, ()>("stake", model_id)
            .expect("stake method")
            .value(amount_wei)
            .send()
            .await
            .expect("stake send failed")
            .await
            .expect("stake confirm failed")
            .expect("stake receipt missing");
    }

    /// Starts a training round for a model.
    pub async fn start_round(&self, model_id: U256, duration_secs: U256) {
        let _receipt: TransactionReceipt = self
            .coordinator
            .method::<_, ()>("startRound", (model_id, duration_secs))
            .expect("startRound method")
            .send()
            .await
            .expect("startRound send failed")
            .await
            .expect("startRound confirm failed")
            .expect("startRound receipt missing");
    }

    /// Submits a proof on-chain.
    /// Returns the transaction receipt for event inspection.
    pub async fn submit_proof(
        &self,
        model_id: U256,
        round_id: U256,
        proof_bytes: Bytes,
        public_inputs: Vec<U256>,
    ) -> TransactionReceipt {
        self.coordinator
            .method::<_, ()>(
                "submitProof",
                (model_id, round_id, proof_bytes, public_inputs),
            )
            .expect("submitProof method")
            .gas(5_000_000u64) // generous gas limit for proof verification
            .send()
            .await
            .expect("submitProof send failed")
            .await
            .expect("submitProof confirm failed")
            .expect("submitProof receipt missing")
    }

    /// Reads the accumulated error bound for a model.
    pub async fn accumulated_error_bound(&self, model_id: U256) -> U256 {
        self.coordinator
            .method::<_, U256>("accumulatedErrorBound", model_id)
            .expect("accumulatedErrorBound method")
            .call()
            .await
            .expect("accumulatedErrorBound call failed")
    }

    /// Reads round data: (modelCommitment, newCommitment, deadline, isCompleted, prover).
    pub async fn get_round(
        &self,
        model_id: U256,
        round_id: U256,
    ) -> (U256, U256, U256, bool, Address) {
        self.coordinator
            .method::<_, (U256, U256, U256, bool, Address)>(
                "rounds",
                (model_id, round_id),
            )
            .expect("rounds method")
            .call()
            .await
            .expect("rounds call failed")
    }

    /// Reads stake data: (amount, lockedUntil, slashed).
    pub async fn get_stake(
        &self,
        prover: Address,
        model_id: U256,
    ) -> (u128, U256, bool) {
        self.coordinator
            .method::<_, (u128, U256, bool)>("stakes", (prover, model_id))
            .expect("stakes method")
            .call()
            .await
            .expect("stakes call failed")
    }

    /// Reads the model's current commitment.
    pub async fn model_commitment(&self, model_id: U256) -> U256 {
        // models() returns (ipfsHash, currentCommitment, owner, currentRound, active, minStake)
        let result: (String, U256, Address, u32, bool, u64) = self
            .coordinator
            .method::<_, (String, U256, Address, u32, bool, u64)>("models", model_id)
            .expect("models method")
            .call()
            .await
            .expect("models call failed");
        result.1 // currentCommitment
    }

    /// Sets the MockVerifier to accept or reject all proofs.
    pub async fn set_mock_accept(&self, accept: bool) {
        let _receipt: TransactionReceipt = self
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

    /// Creates a test environment with the **real** Halo2Verifier.sol deployed.
    ///
    /// Extracts the SRS `[s]₂` G2 point from the Rust prover's KZG parameters
    /// and passes it to the Halo2Verifier constructor. This means the on-chain
    /// verifier will perform real BN254 pairing checks against the same SRS.
    pub async fn new_with_real_verifier(s_g2: [U256; 4]) -> Self {
        let out_dir = ensure_contracts_compiled();

        // Spawn Anvil
        let anvil = Anvil::new().spawn();
        let provider = Provider::<Http>::try_from(anvil.endpoint())
            .expect("Failed to connect to Anvil");

        // Create wallet from first Anvil account
        let wallet: LocalWallet = anvil.keys()[0].clone().into();
        let wallet = wallet.with_chain_id(anvil.chain_id());
        let deployer = wallet.address();
        let client = Arc::new(SignerMiddleware::new(provider, wallet));

        // Deploy real Halo2Verifier with the SRS [s]₂ point
        let (verifier_abi, verifier_bytecode) =
            load_contract_artifact(&out_dir, "Halo2Verifier.sol", "Halo2Verifier");
        let verifier_factory =
            ContractFactory::new(verifier_abi.clone(), verifier_bytecode, client.clone());
        let verifier_contract = verifier_factory
            .deploy(Token::FixedArray(vec![
                Token::Uint(s_g2[0]),
                Token::Uint(s_g2[1]),
                Token::Uint(s_g2[2]),
                Token::Uint(s_g2[3]),
            ]))
            .expect("Halo2Verifier deploy args")
            .send()
            .await
            .expect("Halo2Verifier deploy failed");
        let verifier_addr = verifier_contract.address();

        // Deploy HelixCoordinatorV2(verifier, treasury)
        let (coord_abi, coord_bytecode) =
            load_contract_artifact(&out_dir, "HelixCoordinatorV2.sol", "HelixCoordinatorV2");
        let coord_factory = ContractFactory::new(coord_abi.clone(), coord_bytecode, client.clone());
        let coord_contract = coord_factory
            .deploy((
                Token::Address(verifier_addr),
                Token::Address(deployer),
            ))
            .expect("Coordinator deploy args")
            .send()
            .await
            .expect("Coordinator deploy failed");
        let coordinator_addr = coord_contract.address();

        let verifier = Contract::new(verifier_addr, verifier_abi, client.clone());
        let coordinator = Contract::new(coordinator_addr, coord_abi, client.clone());

        let max_error_bound: U256 = coordinator
            .method::<_, U256>("maxErrorBound", ())
            .expect("maxErrorBound method")
            .call()
            .await
            .expect("maxErrorBound call failed");

        Self {
            anvil,
            client,
            mock_verifier_addr: verifier_addr,
            coordinator_addr,
            mock_verifier: verifier,
            coordinator,
            deployer,
            max_error_bound,
        }
    }

    /// Calls verifyProof directly on the verifier contract (for real verifier tests).
    pub async fn verify_proof_directly(
        &self,
        proof_bytes: Bytes,
        public_inputs: Vec<U256>,
    ) -> bool {
        self.mock_verifier
            .method::<_, bool>("verifyProof", (proof_bytes, public_inputs))
            .expect("verifyProof method")
            .call()
            .await
            .unwrap_or(false)
    }
}

/// Extracts the SRS `[s]₂` G2 point from an `MLTrainingProverV2` as 4 `U256` values
/// suitable for deploying `Halo2Verifier.sol`.
pub fn extract_s_g2_from_prover(
    prover: &helix_prover::MLTrainingProverV2,
) -> [U256; 4] {
    let vk_data = prover.export_vk_data().expect("VK not initialized");
    [
        U256::from_dec_str(&vk_data.s_g2.0).expect("Invalid s_g2[0]"),
        U256::from_dec_str(&vk_data.s_g2.1).expect("Invalid s_g2[1]"),
        U256::from_dec_str(&vk_data.s_g2.2).expect("Invalid s_g2[2]"),
        U256::from_dec_str(&vk_data.s_g2.3).expect("Invalid s_g2[3]"),
    ]
}

// ============================================================================
// Proof Formatting Helpers
// ============================================================================

// ============================================================================
// EvmProofBundle Helper Functions
// ============================================================================

/// Convenience wrapper around `EvmProofBundle` with derived on-chain fields.
///
/// Bridges the canonical `EvmProofBundle` (from helix-prover) to the on-chain
/// test format by pre-computing commitments and extracting typed fields.
pub struct TestEvmProofBundle {
    /// The canonical proof bundle.
    pub bundle: helix_prover::EvmProofBundle,
    /// 8 public inputs as U256 values.
    pub public_inputs: Vec<U256>,
    /// The old state commitment (keccak256 hash pair).
    pub old_commitment: U256,
    /// The new state commitment (keccak256 hash pair).
    pub new_commitment: U256,
    /// Error bound from this step.
    pub error_bound: U256,
    /// Step number.
    pub step_number: U256,
}

impl TestEvmProofBundle {
    /// Creates a test bundle from a `TrainingProofResultV2` and VK data.
    ///
    /// This is the recommended way to create bundles in tests — it uses the
    /// canonical `EvmProofBundle::from_proof_result` internally.
    pub fn from_proof_result_with_vk(
        result: &helix_prover::provers::training_prover_v2::TrainingProofResultV2,
        vk: helix_prover::VkData,
    ) -> Self {
        let bundle = helix_prover::EvmProofBundle::from_proof_result(result, vk)
            .expect("EvmProofBundle creation failed");
        Self::from_evm_bundle(bundle)
    }

    /// Creates a test bundle from a proof result using legacy test signature.
    ///
    /// The `_model_id` and `_max_error_bound` parameters are retained for
    /// backwards compatibility but are no longer used — the canonical
    /// `EvmProofBundle` handles all formatting internally.
    pub fn from_proof_result(
        result: &helix_prover::provers::training_prover_v2::TrainingProofResultV2,
        _model_id: U256,
        _max_error_bound: U256,
    ) -> Self {
        use helix_circuits::verifier::serialize_proof_for_evm;

        // For backwards compat: manually build the EVM proof since we don't
        // have VkData in the legacy call signature.
        let evm_proof = serialize_proof_for_evm(&result.proof, 3)
            .expect("EVM proof serialization failed");

        let evm_pi_bytes = result.to_evm_public_inputs();
        let public_inputs: Vec<U256> = evm_pi_bytes
            .iter()
            .map(|bytes| U256::from_big_endian(bytes))
            .collect();

        assert_eq!(public_inputs.len(), 8, "Expected 8 public inputs");

        let error_bound = public_inputs[5];
        let step_number = public_inputs[6];

        let old_commitment = compute_hash_pair(public_inputs[0], public_inputs[1]);
        let new_commitment = compute_hash_pair(public_inputs[2], public_inputs[3]);

        // Build a minimal EvmProofBundle (without VK data, tests that need
        // VK should use from_proof_result_with_vk instead).
        let bundle = helix_prover::EvmProofBundle {
            evm_proof,
            evm_public_inputs: evm_pi_bytes,
            vk_deployment_args: helix_prover::VkData::default(),
            result: result.clone(),
        };

        Self {
            bundle,
            public_inputs,
            old_commitment,
            new_commitment,
            error_bound,
            step_number,
        }
    }

    /// Wraps an existing `EvmProofBundle` with derived test fields.
    pub fn from_evm_bundle(bundle: helix_prover::EvmProofBundle) -> Self {
        let public_inputs: Vec<U256> = bundle.evm_public_inputs
            .iter()
            .map(|bytes| U256::from_big_endian(bytes))
            .collect();

        let error_bound = public_inputs[5];
        let step_number = public_inputs[6];

        let old_commitment = compute_hash_pair(public_inputs[0], public_inputs[1]);
        let new_commitment = compute_hash_pair(public_inputs[2], public_inputs[3]);

        Self {
            bundle,
            public_inputs,
            old_commitment,
            new_commitment,
            error_bound,
            step_number,
        }
    }

    /// Returns the raw proof bytes.
    pub fn proof_bytes(&self) -> &[u8] {
        &self.bundle.evm_proof
    }

    /// Returns proof bytes as ethers Bytes.
    pub fn proof_as_bytes(&self) -> Bytes {
        Bytes::from(self.bundle.evm_proof.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fr_u256_roundtrip() {
        let fr = Fr::from(12345u64);
        let u = fr_to_u256(&fr);
        assert_eq!(u, U256::from(12345u64));
        let recovered = u256_to_fr(u);
        assert_eq!(fr, recovered);
    }

    #[test]
    fn test_hash_pair_deterministic() {
        let lo = U256::from(1u64);
        let hi = U256::from(2u64);
        let h1 = compute_hash_pair(lo, hi);
        let h2 = compute_hash_pair(lo, hi);
        assert_eq!(h1, h2);
        assert_ne!(h1, U256::zero());
    }

    #[test]
    fn test_hash_pair_different_inputs() {
        let h1 = compute_hash_pair(U256::from(1u64), U256::from(2u64));
        let h2 = compute_hash_pair(U256::from(3u64), U256::from(4u64));
        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }
}
