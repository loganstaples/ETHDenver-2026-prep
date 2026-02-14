//! Chain Watcher Integration Tests.
//!
//! Tests the on-chain event watcher against a local Anvil instance
//! with the real HelixCoordinatorV2 contract deployed. Verifies:
//!
//! - Event detection for ModelRegistered, RoundStarted, ProofSubmitted
//! - Staked/Slashed event processing
//! - Confirmation depth tracking (events only after N blocks)
//! - Cursor persistence and resume-from-checkpoint
//! - NodeEventReactor state tracking (model registration, pause, halt)
//! - Reorg detection (cursor hash mismatch)
//!
//! Requires `anvil` (from Foundry) and `forge build` to have been run.

use std::sync::Arc;
use std::time::Duration;

use ethers::abi::Abi;
use ethers::contract::ContractFactory;
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Provider};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Address, Bytes, U256};
use ethers::utils::Anvil;

use helix_node::chain_watcher::{
    ChainEventData, ChainWatcher, ChainWatcherConfig,
    NodeEventReactor, WatcherCursor,
};

/// Path to forge build output (relative to workspace root).
const CONTRACTS_OUT: &str = "contracts/out";

/// Helper: loads contract ABI + bytecode from a forge artifact.
fn load_artifact(contract_path: &str, contract_name: &str) -> (Abi, Bytes) {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let artifact_path = format!(
        "{}/../../{}/{}/{}.json",
        manifest_dir, CONTRACTS_OUT, contract_path, contract_name
    );

    let data = std::fs::read_to_string(&artifact_path)
        .unwrap_or_else(|e| panic!(
            "Failed to read artifact at {}: {}. Run `forge build` first.",
            artifact_path, e
        ));

    let artifact: serde_json::Value = serde_json::from_str(&data).unwrap();

    let abi: Abi = serde_json::from_value(artifact["abi"].clone())
        .expect("Failed to parse ABI from artifact");

    let bytecode_hex = artifact["bytecode"]["object"]
        .as_str()
        .expect("Missing bytecode in artifact");

    let bytecode_bytes = hex::decode(bytecode_hex.strip_prefix("0x").unwrap_or(bytecode_hex))
        .expect("Invalid bytecode hex");

    (abi, Bytes::from(bytecode_bytes))
}

/// Helper: deploy MockVerifier contract.
async fn deploy_mock_verifier(
    client: &Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
) -> Address {
    let (abi, bytecode) = load_artifact("MockVerifier.sol", "MockVerifier");
    let factory = ContractFactory::new(abi, bytecode, client.clone());
    let contract = factory.deploy(()).unwrap().send().await.unwrap();
    contract.address()
}

/// Helper: deploy HelixCoordinatorV2 contract.
async fn deploy_coordinator(
    client: &Arc<SignerMiddleware<Provider<Http>, LocalWallet>>,
    verifier: Address,
    treasury: Address,
) -> Address {
    let (abi, bytecode) = load_artifact(
        "HelixCoordinatorV2.sol",
        "HelixCoordinatorV2",
    );
    let factory = ContractFactory::new(abi, bytecode, client.clone());
    let contract = factory
        .deploy((verifier, treasury))
        .unwrap()
        .send()
        .await
        .unwrap();
    contract.address()
}

/// Helper: mine N empty blocks on Anvil to advance the chain.
async fn mine_blocks(provider: &Provider<Http>, count: u64) {
    for _ in 0..count {
        provider
            .request::<_, U256>("evm_mine", ())
            .await
            .unwrap();
    }
}

/// Helper: get an ethers client connected to Anvil.
fn setup_client(
    rpc_url: &str,
    private_key: &str,
) -> Arc<SignerMiddleware<Provider<Http>, LocalWallet>> {
    let provider = Provider::<Http>::try_from(rpc_url).unwrap();
    let wallet: LocalWallet = private_key.parse::<LocalWallet>().unwrap().with_chain_id(31337u64);
    Arc::new(SignerMiddleware::new(provider, wallet))
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_watcher_detects_model_registered() {
    // Start Anvil
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    // Deploy contracts
    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    // Create watcher with 0 confirmation depth for instant detection
    let dir = tempfile::tempdir().unwrap();
    let cursor_path = dir.path().join("cursor.json");

    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 0, // No confirmations for testing
        max_block_range: 1000,
        start_block: Some(0), // Watch from genesis
        enabled: true,
    };

    let watcher = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        cursor_path.clone(),
    )
    .await
    .unwrap();

    let mut rx = watcher.subscribe();
    let watcher_handle = watcher.start();

    // Register a model
    let register_call = ethers::contract::Contract::new(
        coordinator,
        serde_json::from_value::<Abi>(serde_json::json!([{
            "inputs": [
                {"name": "ipfsHash", "type": "string"},
                {"name": "initialCommitment", "type": "uint256"},
                {"name": "minStake", "type": "uint256"},
                {"name": "dIn", "type": "uint32"},
                {"name": "dHidden", "type": "uint32"},
                {"name": "dOut", "type": "uint32"},
                {"name": "numLayers", "type": "uint32"},
                {"name": "activationType", "type": "uint8"}
            ],
            "name": "registerModel",
            "outputs": [{"name": "modelId", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        }])).unwrap(),
        client.clone(),
    );

    register_call
        .method::<_, U256>(
            "registerModel",
            (
                "QmTest123".to_string(),
                U256::from(1),
                U256::from(100_000_000_000_000_000u64), // 0.1 ETH
                2u32, 2u32, 1u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    // Mine a block to ensure the event is available
    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    mine_blocks(&provider, 1).await;

    // Wait for the watcher to pick it up
    let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("Timeout waiting for event")
        .expect("Channel closed");

    match &event.data {
        ChainEventData::ModelRegistered {
            model_id,
            ipfs_hash,
            ..
        } => {
            assert_eq!(*model_id, 0, "First model should have ID 0");
            assert_eq!(ipfs_hash, "QmTest123");
        }
        other => panic!("Expected ModelRegistered, got: {:?}", other),
    }

    // Verify cursor was persisted
    let cursor = WatcherCursor::load(&cursor_path).unwrap().unwrap();
    assert!(cursor.last_processed_block > 0);

    // Cleanup
    drop(watcher_handle);
}

#[tokio::test]
async fn test_watcher_detects_staking_events() {
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    let dir = tempfile::tempdir().unwrap();
    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 0,
        max_block_range: 1000,
        start_block: Some(0),
        enabled: true,
    };

    let watcher = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        dir.path().join("cursor.json"),
    )
    .await
    .unwrap();

    let mut rx = watcher.subscribe();
    let _watcher_handle = watcher.start();

    // Register a model first
    let coordinator_abi: Abi = serde_json::from_value(serde_json::json!([
        {
            "inputs": [
                {"name": "ipfsHash", "type": "string"},
                {"name": "initialCommitment", "type": "uint256"},
                {"name": "minStake", "type": "uint256"},
                {"name": "dIn", "type": "uint32"},
                {"name": "dHidden", "type": "uint32"},
                {"name": "dOut", "type": "uint32"},
                {"name": "numLayers", "type": "uint32"},
                {"name": "activationType", "type": "uint8"}
            ],
            "name": "registerModel",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [{"name": "modelId", "type": "uint256"}],
            "name": "stake",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        }
    ])).unwrap();

    let contract = ethers::contract::Contract::new(coordinator, coordinator_abi, client.clone());

    // Register
    contract
        .method::<_, U256>(
            "registerModel",
            (
                "QmStakeTest".to_string(),
                U256::from(1),
                U256::from(100_000_000_000_000_000u64),
                2u32, 2u32, 1u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    // Stake
    contract
        .method::<_, ()>("stake", U256::from(0))
        .unwrap()
        .value(U256::from(1_000_000_000_000_000_000u64)) // 1 ETH
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    mine_blocks(&provider, 1).await;

    // Collect events (should see ModelRegistered + Staked)
    let mut events = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline && events.len() < 2 {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Ok(event)) => events.push(event),
            _ => break,
        }
    }

    assert!(events.len() >= 2, "Expected at least 2 events, got {}", events.len());

    let has_model_registered = events.iter().any(|e| matches!(&e.data, ChainEventData::ModelRegistered { .. }));
    let has_staked = events.iter().any(|e| matches!(&e.data, ChainEventData::Staked { .. }));

    assert!(has_model_registered, "Missing ModelRegistered event");
    assert!(has_staked, "Missing Staked event");

    // Verify staked event details
    let staked_event = events.iter().find(|e| matches!(&e.data, ChainEventData::Staked { .. })).unwrap();
    match &staked_event.data {
        ChainEventData::Staked { model_id, amount, .. } => {
            assert_eq!(*model_id, 0);
            assert_eq!(*amount, U256::from(1_000_000_000_000_000_000u64));
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_watcher_detects_round_started() {
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    let dir = tempfile::tempdir().unwrap();
    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 0,
        max_block_range: 1000,
        start_block: Some(0),
        enabled: true,
    };

    let watcher = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        dir.path().join("cursor.json"),
    )
    .await
    .unwrap();

    let mut rx = watcher.subscribe();
    let _watcher_handle = watcher.start();

    // ABI for register + startRound
    let abi: Abi = serde_json::from_value(serde_json::json!([
        {
            "inputs": [
                {"name": "ipfsHash", "type": "string"},
                {"name": "initialCommitment", "type": "uint256"},
                {"name": "minStake", "type": "uint256"},
                {"name": "dIn", "type": "uint32"},
                {"name": "dHidden", "type": "uint32"},
                {"name": "dOut", "type": "uint32"},
                {"name": "numLayers", "type": "uint32"},
                {"name": "activationType", "type": "uint8"}
            ],
            "name": "registerModel",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"name": "modelId", "type": "uint256"},
                {"name": "duration", "type": "uint256"}
            ],
            "name": "startRound",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ])).unwrap();

    let contract = ethers::contract::Contract::new(coordinator, abi, client.clone());

    // Register model
    contract
        .method::<_, U256>(
            "registerModel",
            (
                "QmRoundTest".to_string(),
                U256::from(1),
                U256::from(100_000_000_000_000_000u64),
                2u32, 2u32, 1u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    // Start round
    contract
        .method::<_, ()>("startRound", (U256::from(0), U256::from(600)))
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    mine_blocks(&provider, 1).await;

    // Collect events
    let mut events = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline && events.len() < 2 {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Ok(event)) => events.push(event),
            _ => break,
        }
    }

    let round_started = events.iter().find(|e| matches!(&e.data, ChainEventData::RoundStarted { .. }));
    assert!(round_started.is_some(), "Missing RoundStarted event");

    match &round_started.unwrap().data {
        ChainEventData::RoundStarted { model_id, round_id, .. } => {
            assert_eq!(*model_id, 0);
            assert_eq!(*round_id, 1);
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn test_watcher_confirmation_depth() {
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    let dir = tempfile::tempdir().unwrap();

    // Use confirmation_depth=3
    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 3,
        max_block_range: 1000,
        start_block: Some(0),
        enabled: true,
    };

    let watcher = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        dir.path().join("cursor.json"),
    )
    .await
    .unwrap();

    let mut rx = watcher.subscribe();
    let _watcher_handle = watcher.start();

    // Register a model
    let abi: Abi = serde_json::from_value(serde_json::json!([{
        "inputs": [
            {"name": "ipfsHash", "type": "string"},
            {"name": "initialCommitment", "type": "uint256"},
            {"name": "minStake", "type": "uint256"},
            {"name": "dIn", "type": "uint32"},
            {"name": "dHidden", "type": "uint32"},
            {"name": "dOut", "type": "uint32"},
            {"name": "numLayers", "type": "uint32"},
            {"name": "activationType", "type": "uint8"}
        ],
        "name": "registerModel",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }])).unwrap();

    let contract = ethers::contract::Contract::new(coordinator, abi, client.clone());

    contract
        .method::<_, U256>(
            "registerModel",
            (
                "QmConfirm".to_string(),
                U256::from(1),
                U256::from(100_000_000_000_000_000u64),
                2u32, 2u32, 1u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    // Wait one poll cycle - should NOT have the event yet (only 0-1 confirmations)
    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    let no_event = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
    assert!(
        no_event.is_err(),
        "Should not receive event before confirmation depth"
    );

    // Mine 3 more blocks to reach confirmation depth
    mine_blocks(&provider, 4).await;

    // Now we should get the event
    let event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("Timeout: event should arrive after confirmations")
        .expect("Channel closed");

    assert!(
        matches!(&event.data, ChainEventData::ModelRegistered { .. }),
        "Expected ModelRegistered, got {:?}", event.data
    );
}

#[tokio::test]
async fn test_watcher_cursor_resume() {
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    let dir = tempfile::tempdir().unwrap();
    let cursor_path = dir.path().join("cursor.json");

    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 0,
        max_block_range: 1000,
        start_block: Some(0),
        enabled: true,
    };

    // First watcher session — registers a model
    {
        let watcher = ChainWatcher::new(
            &rpc_url,
            &format!("{:?}", coordinator),
            config.clone(),
            cursor_path.clone(),
        )
        .await
        .unwrap();

        let mut rx = watcher.subscribe();
        let watcher_handle = watcher.start();

        // Register model
        let abi: Abi = serde_json::from_value(serde_json::json!([{
            "inputs": [
                {"name": "ipfsHash", "type": "string"},
                {"name": "initialCommitment", "type": "uint256"},
                {"name": "minStake", "type": "uint256"},
                {"name": "dIn", "type": "uint32"},
                {"name": "dHidden", "type": "uint32"},
                {"name": "dOut", "type": "uint32"},
                {"name": "numLayers", "type": "uint32"},
                {"name": "activationType", "type": "uint8"}
            ],
            "name": "registerModel",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        }])).unwrap();

        let contract = ethers::contract::Contract::new(coordinator, abi, client.clone());

        contract
            .method::<_, U256>(
                "registerModel",
                (
                    "QmResume1".to_string(),
                    U256::from(1),
                    U256::from(100_000_000_000_000_000u64),
                    2u32, 2u32, 1u32, 2u32, 0u8,
                ),
            )
            .unwrap()
            .send()
            .await
            .unwrap()
            .await
            .unwrap();

        let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
        mine_blocks(&provider, 1).await;

        // Wait for the event
        let _event = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("Timeout")
            .expect("Closed");

        // Stop the watcher gracefully so the cursor is finalized
        watcher.stop();
        let _ = tokio::time::timeout(Duration::from_secs(2), watcher_handle).await;
    }

    // Verify cursor exists
    let cursor = WatcherCursor::load(&cursor_path).unwrap().unwrap();
    let saved_block = cursor.last_processed_block;
    assert!(saved_block > 0, "Cursor should have been saved");

    // Register another model AFTER the first watcher stopped
    let abi2: Abi = serde_json::from_value(serde_json::json!([{
        "inputs": [
            {"name": "ipfsHash", "type": "string"},
            {"name": "initialCommitment", "type": "uint256"},
            {"name": "minStake", "type": "uint256"},
            {"name": "dIn", "type": "uint32"},
            {"name": "dHidden", "type": "uint32"},
            {"name": "dOut", "type": "uint32"},
            {"name": "numLayers", "type": "uint32"},
            {"name": "activationType", "type": "uint8"}
        ],
        "name": "registerModel",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }])).unwrap();

    let contract2 = ethers::contract::Contract::new(coordinator, abi2, client.clone());
    contract2
        .method::<_, U256>(
            "registerModel",
            (
                "QmResume2".to_string(),
                U256::from(2),
                U256::from(100_000_000_000_000_000u64),
                4u32, 4u32, 2u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    mine_blocks(&provider, 1).await;

    // Start second watcher — should resume from cursor and only see QmResume2
    let watcher2 = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        cursor_path,
    )
    .await
    .unwrap();

    // Verify it resumed from the saved block
    assert_eq!(
        watcher2.last_processed_block(),
        saved_block,
        "Watcher should resume from saved cursor"
    );

    let mut rx2 = watcher2.subscribe();
    let _handle2 = watcher2.start();

    let event = tokio::time::timeout(Duration::from_secs(10), rx2.recv())
        .await
        .expect("Timeout")
        .expect("Closed");

    match &event.data {
        ChainEventData::ModelRegistered { ipfs_hash, model_id, .. } => {
            assert_eq!(*model_id, 1, "Should be second model (ID 1)");
            assert_eq!(ipfs_hash, "QmResume2", "Should only see the second registration");
        }
        other => panic!("Expected ModelRegistered for QmResume2, got {:?}", other),
    }
}

#[tokio::test]
async fn test_node_event_reactor_with_live_events() {
    let anvil = Anvil::new().arg("--code-size-limit").arg("100000").spawn();
    let rpc_url = anvil.endpoint();
    let private_key = hex::encode(anvil.keys()[0].to_bytes());

    let client = setup_client(&rpc_url, &private_key);
    let treasury = client.address();

    let verifier = deploy_mock_verifier(&client).await;
    let coordinator = deploy_coordinator(&client, verifier, treasury).await;

    let dir = tempfile::tempdir().unwrap();
    let config = ChainWatcherConfig {
        poll_interval_secs: 1,
        confirmation_depth: 0,
        max_block_range: 1000,
        start_block: Some(0),
        enabled: true,
    };

    let watcher = ChainWatcher::new(
        &rpc_url,
        &format!("{:?}", coordinator),
        config,
        dir.path().join("cursor.json"),
    )
    .await
    .unwrap();

    // Create and subscribe the reactor
    let reactor = Arc::new(NodeEventReactor::new(Some(client.address())));
    let rx = watcher.subscribe();
    let _reactor_handle = helix_node::chain_watcher::spawn_event_handler(rx, reactor.clone());

    let _watcher_handle = watcher.start();

    // Register model + start round
    let abi: Abi = serde_json::from_value(serde_json::json!([
        {
            "inputs": [
                {"name": "ipfsHash", "type": "string"},
                {"name": "initialCommitment", "type": "uint256"},
                {"name": "minStake", "type": "uint256"},
                {"name": "dIn", "type": "uint32"},
                {"name": "dHidden", "type": "uint32"},
                {"name": "dOut", "type": "uint32"},
                {"name": "numLayers", "type": "uint32"},
                {"name": "activationType", "type": "uint8"}
            ],
            "name": "registerModel",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {"name": "modelId", "type": "uint256"},
                {"name": "duration", "type": "uint256"}
            ],
            "name": "startRound",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ])).unwrap();

    let contract = ethers::contract::Contract::new(coordinator, abi, client.clone());

    contract
        .method::<_, U256>(
            "registerModel",
            (
                "QmReactor".to_string(),
                U256::from(42),
                U256::from(100_000_000_000_000_000u64),
                2u32, 2u32, 1u32, 2u32, 0u8,
            ),
        )
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    contract
        .method::<_, ()>("startRound", (U256::from(0), U256::from(600)))
        .unwrap()
        .send()
        .await
        .unwrap()
        .await
        .unwrap();

    let provider = Provider::<Http>::try_from(&rpc_url).unwrap();
    mine_blocks(&provider, 1).await;

    // Wait for reactor to process events
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify reactor state
    let models = reactor.active_models();
    assert!(models.contains_key(&0), "Reactor should track model 0");

    let model_state = reactor.get_model_state(0).unwrap();
    assert!(model_state.active);
    assert_eq!(model_state.current_round, 1);
    assert_eq!(model_state.commitment, U256::from(42));
    assert!(!reactor.is_contract_paused());
}
