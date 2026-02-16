//! End-to-end tests for the optional ZK proof layer.
//!
//! These tests verify that:
//! 1. ZK proofs can be generated from real MPC training output
//! 2. Proofs are absent when ZK is disabled (zero overhead)
//! 3. Checkpoint frequency filtering works correctly
//! 4. Proof format is correct (real KZG, EVM-compatible)
//! 5. Invalid proofs are rejected by the prover's self-verification
//!
//! Note: On-chain verification tests require the `chain` feature and Foundry.
//! The pure Rust tests (proof generation, format, frequency) run without chain.

use helix_client::zk_proof_layer::{ZkProofConfig, ZkProofLayer};
use helix_mpc::e2e_integration::FinalWeights;
use helix_prover::halo2curves::bn256::Fr as Halo2Fr;
use helix_prover::halo2curves::ff::PrimeField;

// ============================================================================
// Helpers
// ============================================================================

/// Creates a small test model (4→2→3) with distinct weight sets for each
/// checkpoint to simulate training progression.
fn make_weights(scale: f64) -> FinalWeights {
    let d_in = 4;
    let d_hid = 2;
    let d_out = 3;

    FinalWeights {
        w1: (0..d_hid * d_in)
            .map(|i| (i as f64 + 1.0) * 0.01 * scale)
            .collect(),
        b1: vec![0.001 * scale; d_hid],
        w2: (0..d_out * d_hid)
            .map(|i| (i as f64 + 1.0) * 0.02 * scale)
            .collect(),
        b2: vec![0.002 * scale; d_out],
    }
}

/// Number of weights for a 4→2→3 model.
const NUM_WEIGHTS: usize = 4 * 2 + 2 + 2 * 3 + 3; // = 17

// ============================================================================
// Test 1: ZK Proof E2E
// ============================================================================

/// Enable ZK, run simulated MPC training steps with checkpoints, generate
/// proofs for checkpoint transitions, and verify them.
#[test]
fn test_zk_proof_e2e() {
    let config = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 1, // Prove every checkpoint
        self_verify: true,
    };
    let mut layer = ZkProofLayer::new(config, 4, 2, 3);

    // Simulate 6 checkpoints (steps 5, 10, 15, 20, 25, 30)
    let weights: Vec<FinalWeights> = (1..=6).map(|i| make_weights(i as f64)).collect();

    let mut proofs_generated = 0;

    for (idx, w) in weights.iter().enumerate() {
        let step = (idx + 1) * 5;
        let loss = 2.5 - (idx as f64 * 0.3); // Decreasing loss

        let result = layer
            .process_checkpoint(idx, step, w, loss)
            .expect("process_checkpoint should not error");

        if idx == 0 {
            // First checkpoint is baseline — no proof generated
            assert!(
                result.is_none(),
                "First checkpoint should be baseline (no proof)"
            );
        } else {
            // All subsequent checkpoints should produce proofs
            assert!(
                result.is_some(),
                "Checkpoint {} should produce a ZK proof",
                idx
            );
            let zk_result = result.unwrap();
            assert_eq!(zk_result.step, step);
            assert!(
                zk_result.proof.proof_size() > 32,
                "Proof must be a real KZG proof (>32 bytes), got {} bytes",
                zk_result.proof.proof_size()
            );
            assert!(zk_result.proof.verified, "Proof must self-verify");
            assert_eq!(
                zk_result.proof.public_inputs.len(),
                6,
                "StateTransitionCircuit must produce 6 public inputs"
            );
            proofs_generated += 1;
        }
    }

    assert_eq!(proofs_generated, 5, "Should generate 5 proofs for 6 checkpoints");
    assert_eq!(layer.proofs_generated(), 5);
    assert_eq!(layer.stats().proofs_verified, 5);
    assert!(
        layer.stats().total_proving_time.as_millis() > 0,
        "Total proving time must be positive"
    );
}

// ============================================================================
// Test 2: ZK Disabled — No Overhead
// ============================================================================

/// Verify that when ZK is disabled, no prover code is invoked and no
/// resources are allocated.
#[test]
fn test_zk_disabled_no_overhead() {
    let config = ZkProofConfig {
        enabled: false,
        checkpoint_frequency: 1,
        self_verify: true,
    };
    let mut layer = ZkProofLayer::new(config, 4, 2, 3);

    // Process the same 6 checkpoints as the E2E test
    let weights: Vec<FinalWeights> = (1..=6).map(|i| make_weights(i as f64)).collect();

    for (idx, w) in weights.iter().enumerate() {
        let step = (idx + 1) * 5;
        let loss = 2.5 - (idx as f64 * 0.3);

        let result = layer
            .process_checkpoint(idx, step, w, loss)
            .expect("process_checkpoint should not error");

        assert!(
            result.is_none(),
            "ZK disabled: checkpoint {} should return None",
            idx
        );
    }

    // Verify absolutely no prover state was allocated
    assert_eq!(layer.proofs_generated(), 0);
    assert!(!layer.is_enabled());
    assert_eq!(layer.stats().total_proving_time.as_millis(), 0);
    assert!(layer.stats().init_time.is_none(), "Prover should never be initialized");
}

// ============================================================================
// Test 3: ZK Checkpoint Frequency
// ============================================================================

/// ZK freq=2, 7 checkpoints (0-6): baseline at 0, proofs at indices 1,3,5
/// (because (idx+1) % 2 == 0 for idx=1,3,5). Verify exactly 3 ZK proofs.
#[test]
fn test_zk_checkpoint_frequency() {
    let config = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 2, // Prove every 2nd checkpoint
        self_verify: false,      // Skip self-verify for speed
    };
    let mut layer = ZkProofLayer::new(config, 4, 2, 3);

    let weights: Vec<FinalWeights> = (1..=7).map(|i| make_weights(i as f64)).collect();
    let mut proof_indices = Vec::new();

    for (idx, w) in weights.iter().enumerate() {
        let step = (idx + 1) * 10;
        let loss = 3.0 - (idx as f64 * 0.3);

        let result = layer
            .process_checkpoint(idx, step, w, loss)
            .expect("process_checkpoint should not error");

        if let Some(zk_result) = result {
            proof_indices.push(idx);
            assert_eq!(zk_result.step, step);
            assert!(zk_result.proof.proof_size() > 32);
        }
    }

    // Checkpoint 0 = baseline (no proof)
    // Checkpoint 1: (1+1)%2 = 0 → proof (transition 0→1)
    // Checkpoint 2: (2+1)%2 = 1 → skip
    // Checkpoint 3: (3+1)%2 = 0 → proof (transition 1→3... but actually from prev_weights)
    // Checkpoint 4: (4+1)%2 = 1 → skip
    // Checkpoint 5: (5+1)%2 = 0 → proof
    // Checkpoint 6: (6+1)%2 = 1 → skip
    assert_eq!(
        proof_indices,
        vec![1, 3, 5],
        "Proofs should be generated at indices 1, 3, 5 for freq=2"
    );
    assert_eq!(layer.proofs_generated(), 3, "Should generate exactly 3 proofs");
}

// ============================================================================
// Test 4: ZK Proof Format
// ============================================================================

/// Generate a proof and verify it has the correct format for EVM submission:
/// - Proof bytes: multiple of 32 bytes (field elements)
/// - Public inputs: 6 elements, each convertible to 32-byte big-endian uint256
/// - Proof size consistent with SHPLONK KZG proofs
#[test]
fn test_zk_proof_format() {
    let config = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 1,
        self_verify: true,
    };
    let mut layer = ZkProofLayer::new(config, 4, 2, 3);

    let weights_v1 = make_weights(1.0);
    let weights_v2 = make_weights(2.0);

    // Baseline
    layer
        .process_checkpoint(0, 10, &weights_v1, 2.0)
        .unwrap();

    // Generate proof
    let result = layer
        .process_checkpoint(1, 20, &weights_v2, 1.5)
        .unwrap()
        .expect("Should produce a proof");

    let proof = &result.proof;

    // --- Proof bytes format ---
    assert!(
        !proof.proof_bytes.is_empty(),
        "Proof bytes must be non-empty"
    );
    assert_eq!(
        proof.proof_bytes.len() % 32,
        0,
        "Proof size must be a multiple of 32 bytes (field element size), got {} bytes",
        proof.proof_bytes.len()
    );
    // SHPLONK KZG proofs are typically 320-3000 bytes
    assert!(
        proof.proof_size() >= 128,
        "Proof too small for real KZG ({} bytes)",
        proof.proof_size()
    );
    assert!(
        proof.proof_size() < 100_000,
        "Proof too large ({} bytes), likely corrupt",
        proof.proof_size()
    );

    // --- Public inputs format ---
    assert_eq!(
        proof.public_inputs.len(),
        6,
        "StateTransitionCircuit produces 6 public inputs"
    );

    // Verify public inputs are valid field elements (< BN254 scalar order)
    let modulus_bytes = [
        0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29,
        0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
        0x97, 0x81, 0x6a, 0x91, 0x68, 0x71, 0xca, 0x8d,
        0x3c, 0x20, 0x8c, 0x16, 0xd8, 0x7c, 0xfd, 0x47,
    ];
    for (i, pi) in proof.public_inputs.iter().enumerate() {
        let repr = pi.to_repr();
        let bytes: &[u8] = repr.as_ref();
        assert_eq!(bytes.len(), 32, "PI {} repr must be 32 bytes", i);
        // Verify it's less than the modulus (most significant byte check)
        // The modulus starts with 0x30, so if MSB > 0x30, it would be invalid
        // This is a basic sanity check, not a full comparison
        assert!(
            bytes[31] <= 0x30,
            "PI {} may exceed field modulus (MSB = 0x{:02x})",
            i,
            bytes[31]
        );
    }

    // --- EVM format ---
    let evm_proof = proof.to_evm_proof();
    assert_eq!(
        evm_proof.len(),
        proof.proof_bytes.len(),
        "EVM proof should be same bytes as raw proof"
    );

    let evm_pis = proof.to_evm_public_inputs();
    assert_eq!(evm_pis.len(), 6);
    for (i, pi_bytes) in evm_pis.iter().enumerate() {
        assert_eq!(pi_bytes.len(), 32, "EVM PI {} must be 32 bytes", i);
    }

    // Verify the proof was self-verified during generation
    assert!(proof.verified, "Proof must be self-verified");

    // Verify generation time is reasonable (> 0, < 60s)
    assert!(
        proof.generation_time.as_millis() > 0,
        "Generation time must be positive"
    );
    assert!(
        proof.generation_time.as_secs() < 60,
        "Generation time should be < 60s, got {}s",
        proof.generation_time.as_secs()
    );
}

// ============================================================================
// Test 5: Invalid Proof Rejected
// ============================================================================

/// Generate a valid proof, then modify it to create an invalid proof.
/// Verify the prover's verification rejects it.
#[test]
fn test_invalid_proof_rejected() {
    use helix_circuits::ml::state_transition::StateTransitionWitness;
    use helix_prover::{CheckpointProver, CheckpointProverConfig};

    let num_weights = 8;

    // Create valid weights
    let old_weights: Vec<Halo2Fr> = (0..num_weights)
        .map(|i| Halo2Fr::from((i + 1) as u64))
        .collect();
    let new_weights: Vec<Halo2Fr> = (0..num_weights)
        .map(|i| Halo2Fr::from((i + 2) as u64))
        .collect();

    let witness = StateTransitionWitness::new(old_weights, new_weights, Halo2Fr::from(100u64));

    // Create prover
    let config = CheckpointProverConfig::minimal(num_weights);
    let prover = CheckpointProver::new(config);

    // Generate a valid proof
    let result = prover.prove(&witness).expect("Valid proof should succeed");
    assert!(
        prover.verify(&result).expect("Verification should not error"),
        "Valid proof must verify"
    );

    // --- Tamper with proof bytes (flip a byte) ---
    let mut tampered_proof = result.clone();
    if !tampered_proof.proof_bytes.is_empty() {
        // Flip a byte in the middle of the proof
        let mid = tampered_proof.proof_bytes.len() / 2;
        tampered_proof.proof_bytes[mid] ^= 0xFF;
    }
    let tampered_valid = prover
        .verify(&tampered_proof)
        .expect("Verification should not error even for invalid proof");
    assert!(
        !tampered_valid,
        "Tampered proof must NOT verify"
    );

    // --- Wrong public inputs ---
    let mut wrong_pi_result = result.clone();
    wrong_pi_result.public_inputs[0] = Halo2Fr::from(99999u64); // Wrong old_hash_lo
    let wrong_pi_valid = prover
        .verify(&wrong_pi_result)
        .expect("Verification should not error");
    assert!(
        !wrong_pi_valid,
        "Proof with wrong public inputs must NOT verify"
    );

    // --- Empty proof bytes ---
    let mut empty_proof = result.clone();
    empty_proof.proof_bytes = vec![];
    let empty_result = prover.verify(&empty_proof);
    // Empty proof should either error or return false
    match empty_result {
        Ok(valid) => assert!(!valid, "Empty proof must not verify"),
        Err(_) => {} // Error is also acceptable for empty proof
    }

    // --- Truncated proof ---
    let mut truncated_proof = result.clone();
    truncated_proof.proof_bytes = result.proof_bytes[..32].to_vec();
    let truncated_result = prover.verify(&truncated_proof);
    match truncated_result {
        Ok(valid) => assert!(!valid, "Truncated proof must not verify"),
        Err(_) => {} // Error is acceptable for malformed proof
    }

    // --- Garbage proof of correct length ---
    let mut garbage_proof = result.clone();
    garbage_proof.proof_bytes = vec![0xAB; result.proof_bytes.len()];
    let garbage_result = prover.verify(&garbage_proof);
    match garbage_result {
        Ok(valid) => assert!(!valid, "Garbage proof must not verify"),
        Err(_) => {} // Error is acceptable for garbage data
    }
}

// ============================================================================
// Additional Tests: Incremental Proving Correctness
// ============================================================================

/// Verify that incremental proofs correctly track weight transitions:
/// each proof's old_hash matches the previous proof's new_hash.
#[test]
fn test_zk_incremental_proving_consistency() {
    let config = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 1,
        self_verify: true,
    };
    let mut layer = ZkProofLayer::new(config, 4, 2, 3);

    let weights: Vec<FinalWeights> = (1..=4).map(|i| make_weights(i as f64)).collect();
    let mut proofs = Vec::new();

    for (idx, w) in weights.iter().enumerate() {
        let step = (idx + 1) * 10;
        let result = layer.process_checkpoint(idx, step, w, 2.0 - idx as f64 * 0.3).unwrap();
        if let Some(zk_result) = result {
            proofs.push(zk_result);
        }
    }

    assert_eq!(proofs.len(), 3, "Should have 3 proofs from 4 checkpoints");

    // Verify incremental consistency: proof[i].new_hash == proof[i+1].old_hash
    for i in 0..proofs.len() - 1 {
        let current_new_hash_lo = proofs[i].proof.public_inputs[2]; // new_hash_lo
        let current_new_hash_hi = proofs[i].proof.public_inputs[3]; // new_hash_hi
        let next_old_hash_lo = proofs[i + 1].proof.public_inputs[0]; // old_hash_lo
        let next_old_hash_hi = proofs[i + 1].proof.public_inputs[1]; // old_hash_hi

        assert_eq!(
            current_new_hash_lo, next_old_hash_lo,
            "Proof {} new_hash_lo must equal proof {} old_hash_lo",
            i, i + 1
        );
        assert_eq!(
            current_new_hash_hi, next_old_hash_hi,
            "Proof {} new_hash_hi must equal proof {} old_hash_hi",
            i, i + 1
        );
    }
}

/// Verify that the ZK layer correctly handles the config fields being
/// wired through the FullOrchestrationConfig.
#[test]
fn test_zk_config_wiring() {
    use helix_client::full_orchestration::FullOrchestrationConfig;

    let config = FullOrchestrationConfig::default();
    // Default: ZK disabled
    assert!(!config.zk_proof.enabled);
    assert_eq!(config.zk_proof.checkpoint_frequency, 1);

    // Verify ZK config can be set
    let mut config = FullOrchestrationConfig::default();
    config.zk_proof = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 3,
        self_verify: false,
    };
    assert!(config.zk_proof.enabled);
    assert_eq!(config.zk_proof.checkpoint_frequency, 3);
}

/// Verify that ZK proof generation works with a larger model size
/// (closer to the real MNIST 784→32→10 architecture, but using a
/// smaller version to keep tests fast).
#[test]
fn test_zk_proof_larger_model() {
    // 16→8→4 model (160 weights — larger than minimal but still fast)
    let d_in = 16;
    let d_hid = 8;
    let d_out = 4;
    let num_weights = d_in * d_hid + d_hid + d_hid * d_out + d_out; // = 168

    let config = ZkProofConfig {
        enabled: true,
        checkpoint_frequency: 1,
        self_verify: true,
    };
    let mut layer = ZkProofLayer::new(config, d_in, d_hid, d_out);

    let weights_v1 = FinalWeights {
        w1: vec![0.01; d_in * d_hid],
        b1: vec![0.0; d_hid],
        w2: vec![0.02; d_hid * d_out],
        b2: vec![0.0; d_out],
    };
    let weights_v2 = FinalWeights {
        w1: vec![0.015; d_in * d_hid],
        b1: vec![0.001; d_hid],
        w2: vec![0.025; d_hid * d_out],
        b2: vec![0.001; d_out],
    };

    // Baseline
    let result = layer.process_checkpoint(0, 10, &weights_v1, 2.0).unwrap();
    assert!(result.is_none());

    // Generate proof
    let result = layer.process_checkpoint(1, 20, &weights_v2, 1.5).unwrap();
    assert!(result.is_some(), "Should produce a proof for larger model");

    let zk_result = result.unwrap();
    assert!(zk_result.proof.proof_size() > 32);
    assert!(zk_result.proof.verified);
    assert_eq!(zk_result.proof.public_inputs.len(), 6);
}

// ============================================================================
// Chain Integration Tests (require `chain` feature + Foundry)
// ============================================================================

#[cfg(feature = "chain")]
mod chain_tests {
    use super::*;
    use std::process::{Child, Command, Stdio};
    use std::str::FromStr;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ethers::prelude::*;
    use ethers::providers::{Http, Provider};
    use ethers::signers::{LocalWallet, Signer};
    use ethers::types::{Address, U256};

    use helix_client::rpc::chain_v4::{ChainClientV4, Halo2VerifierContract};

    const DEPLOYER_KEY: &str =
        "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const WORKER1_KEY: &str =
        "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
    const WORKER2_KEY: &str =
        "5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";
    const CHAIN_ID: u64 = 31337;

    struct AnvilInstance {
        child: Child,
        rpc_url: String,
    }

    impl AnvilInstance {
        fn start() -> Self {
            let port = portpicker::pick_unused_port().unwrap_or(8599);
            let rpc_url = format!("http://127.0.0.1:{}", port);

            let child = Command::new("anvil")
                .args([
                    "--port",
                    &port.to_string(),
                    "--accounts",
                    "10",
                    "--balance",
                    "10000",
                    "--silent",
                    "--code-size-limit",
                    "100000",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("Failed to start anvil (is Foundry installed?)");

            // Wait for anvil to be ready
            let deadline = Instant::now() + Duration::from_secs(10);
            let addr: std::net::SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
            loop {
                if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok()
                {
                    break;
                }
                if Instant::now() > deadline {
                    panic!("Anvil failed to start within 10 seconds");
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            Self { child, rpc_url }
        }
    }

    impl Drop for AnvilInstance {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    /// Deploy V4 coordinator with real Halo2Verifier, submit a ZK proof
    /// checkpoint, and verify it passes on-chain verification.
    #[tokio::test]
    async fn test_zk_proof_on_chain_with_real_verifier() {
        let anvil = AnvilInstance::start();

        let provider = Provider::<Http>::try_from(&anvil.rpc_url).unwrap();
        let deployer_wallet = LocalWallet::from_str(DEPLOYER_KEY)
            .unwrap()
            .with_chain_id(CHAIN_ID);
        let client = Arc::new(SignerMiddleware::new(provider.clone(), deployer_wallet.clone()));

        // Deploy the real Halo2Verifier
        let verifier_contract = Halo2VerifierContract::deploy(client.clone(), ())
            .expect("Halo2Verifier deploy prepare")
            .send()
            .await
            .expect("Halo2Verifier deploy send");
        let verifier_address = verifier_contract.address();

        // Deploy V4 coordinator with the real verifier
        let treasury = deployer_wallet.address();
        let (chain_client, _deploy_result) = ChainClientV4::deploy(
            &anvil.rpc_url,
            DEPLOYER_KEY,
            treasury,
            verifier_address,
            Some(CHAIN_ID),
        )
        .await
        .expect("V4 deploy should succeed");

        // Register a job
        let payment = ethers::utils::parse_ether(1.0).unwrap();
        let (_, job_id) = chain_client
            .register_training_job([0xAA; 32], 10, 5, payment)
            .await
            .expect("Job registration should succeed");

        // Workers stake and join
        for key in [WORKER1_KEY, WORKER2_KEY] {
            let wallet = LocalWallet::from_str(key).unwrap().with_chain_id(CHAIN_ID);
            let worker_client = ChainClientV4::with_wallet(
                &anvil.rpc_url,
                wallet,
                &format!("{:?}", chain_client.coordinator_address()),
            )
            .await
            .unwrap();
            let stake = ethers::utils::parse_ether(0.1).unwrap();
            worker_client.stake_and_join(job_id, stake).await.unwrap();
        }

        // Generate a real ZK proof using the ZkProofLayer
        let zk_config = ZkProofConfig {
            enabled: true,
            checkpoint_frequency: 1,
            self_verify: true,
        };
        let mut zk_layer = ZkProofLayer::new(zk_config, 4, 2, 3);

        let weights_v1 = make_weights(1.0);
        let weights_v2 = make_weights(2.0);

        // Baseline
        zk_layer
            .process_checkpoint(0, 10, &weights_v1, 2.0)
            .unwrap();

        // Generate proof
        let zk_result = zk_layer
            .process_checkpoint(1, 20, &weights_v2, 1.5)
            .unwrap()
            .expect("Should generate a ZK proof");

        // Prepare for on-chain submission
        let evm_proof = zk_result.proof.to_evm_proof();
        let evm_public_inputs: Vec<U256> = zk_result
            .proof
            .to_evm_public_inputs()
            .iter()
            .map(|bytes| U256::from_big_endian(bytes))
            .collect();

        // Submit checkpoint with ZK proof
        let receipt = chain_client
            .submit_checkpoint_with_proof(
                job_id,
                20,                 // step
                [0xBB; 32],        // weight commitment (arbitrary for test)
                U256::from(1500),  // loss
                evm_proof,
                evm_public_inputs,
            )
            .await
            .expect("submitCheckpointWithProof should succeed with real verifier");

        assert!(
            receipt.status.map(|s| s.as_u64()) == Some(1),
            "Transaction must succeed (status = 1)"
        );

        // Verify checkpoint was recorded
        let cp = chain_client.get_checkpoint(job_id, 0).await.unwrap();
        assert_eq!(cp.step_number, 20);
        assert_eq!(cp.signer_count, 0, "ZK path should have signer_count=0");
    }

    /// Submit a garbage proof to the on-chain verifier and verify rejection.
    #[tokio::test]
    async fn test_invalid_proof_rejected_on_chain() {
        let anvil = AnvilInstance::start();

        let provider = Provider::<Http>::try_from(&anvil.rpc_url).unwrap();
        let deployer_wallet = LocalWallet::from_str(DEPLOYER_KEY)
            .unwrap()
            .with_chain_id(CHAIN_ID);
        let client = Arc::new(SignerMiddleware::new(provider.clone(), deployer_wallet.clone()));

        // Deploy real Halo2Verifier + V4 coordinator
        let verifier_contract = Halo2VerifierContract::deploy(client.clone(), ())
            .expect("deploy prepare")
            .send()
            .await
            .expect("deploy send");
        let verifier_address = verifier_contract.address();

        let treasury = deployer_wallet.address();
        let (chain_client, _) = ChainClientV4::deploy(
            &anvil.rpc_url,
            DEPLOYER_KEY,
            treasury,
            verifier_address,
            Some(CHAIN_ID),
        )
        .await
        .expect("V4 deploy");

        // Register job and add workers
        let payment = ethers::utils::parse_ether(1.0).unwrap();
        let (_, job_id) = chain_client
            .register_training_job([0xAA; 32], 10, 5, payment)
            .await
            .unwrap();

        for key in [WORKER1_KEY, WORKER2_KEY] {
            let wallet = LocalWallet::from_str(key).unwrap().with_chain_id(CHAIN_ID);
            let worker_client = ChainClientV4::with_wallet(
                &anvil.rpc_url,
                wallet,
                &format!("{:?}", chain_client.coordinator_address()),
            )
            .await
            .unwrap();
            let stake = ethers::utils::parse_ether(0.1).unwrap();
            worker_client.stake_and_join(job_id, stake).await.unwrap();
        }

        // Submit garbage proof — should be rejected
        let garbage_proof = vec![0xAB; 1856]; // Correct length but garbage data
        let garbage_pis = vec![U256::from(1); 6]; // 6 public inputs

        let result = chain_client
            .submit_checkpoint_with_proof(
                job_id,
                10,
                [0xCC; 32],
                U256::from(500),
                garbage_proof,
                garbage_pis,
            )
            .await;

        assert!(
            result.is_err(),
            "Garbage proof must be rejected by the on-chain Halo2Verifier"
        );
    }
}
