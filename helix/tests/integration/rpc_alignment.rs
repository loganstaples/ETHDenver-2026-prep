//! Integration test: Client ↔ Node RPC Protocol Alignment
//!
//! Starts a node RPC server in-process, creates a `HelixRpcClient` pointing at
//! it, and verifies every RPC method deserializes into the expected client type.

use std::net::SocketAddr;

use helix_client::rpc::{
    GenerateProofAck, HealthStatus, HelixRpcClient, HelixRpcConfig, ModelInfo, NetworkStatus,
    NodeCapabilities, ProofStatus, ProofSubmission, RoundInfo, StakingInfo, TrainingProgress,
    TrainingResultData, TrainingStatus, WorkerInfo,
};
use helix_node::api::rpc::{create_default_rpc_state, start_rpc_server};

/// Find a free TCP port by binding to port 0 and reading the assigned port.
async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Boot the node RPC server in a background task and return a configured client.
async fn setup() -> HelixRpcClient {
    let port = free_port().await;
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    let state = create_default_rpc_state("aggregator", &addr.to_string());

    // Set an active model so handlers have data to work with
    *state.model_id.write() = Some(1);

    // Subscribe to triggers so sends don't fail with "no listeners"
    let _round_rx = state.round_trigger_tx.subscribe();
    let _stop_rx = state.stop_trigger_tx.subscribe();

    // Leak the receivers so they live for the duration of the test
    let round_rx = Box::new(_round_rx);
    let stop_rx = Box::new(_stop_rx);
    std::mem::forget(round_rx);
    std::mem::forget(stop_rx);

    let server_state = state.clone();
    tokio::spawn(async move {
        start_rpc_server(addr, server_state).await.unwrap();
    });

    // Wait briefly for the server to start
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let config = HelixRpcConfig::with_endpoint(&format!("http://127.0.0.1:{}/rpc", port));
    HelixRpcClient::new(config).unwrap()
}

#[tokio::test]
async fn test_connect_health_and_capabilities() {
    let client = setup().await;

    // connect() calls helix_health + helix_capabilities
    client.connect().await.expect("connect() should succeed");
    assert!(client.is_connected().await);
}

#[tokio::test]
async fn test_health_check_shape() {
    let client = setup().await;
    let health: HealthStatus = client.health_check().await.expect("health_check failed");
    assert!(health.healthy);
    assert!(!health.version.is_empty());
}

#[tokio::test]
async fn test_capabilities_shape() {
    let client = setup().await;
    let caps: NodeCapabilities = client.get_capabilities().await.expect("capabilities failed");
    assert!(caps.can_prove);
    assert!(caps.can_verify);
    assert!(caps.can_aggregate); // aggregator role
}

#[tokio::test]
async fn test_training_status_shape() {
    let client = setup().await;
    let status: TrainingStatus = client
        .get_training_status()
        .await
        .expect("training_status failed");
    // No active training
    assert!(!status.active);
    assert_eq!(status.model_id, 1);
}

#[tokio::test]
async fn test_network_status_shape() {
    let client = setup().await;
    let net: NetworkStatus = client
        .get_network_status()
        .await
        .expect("network_status failed");
    assert_eq!(net.chain_id, 31337);
}

#[tokio::test]
async fn test_proof_status_shape() {
    let client = setup().await;
    let proof: ProofStatus = client.get_proof_status().await.expect("proof_status failed");
    assert!(!proof.generating);
}

#[tokio::test]
async fn test_workers_shape() {
    let client = setup().await;
    let workers: Vec<WorkerInfo> = client.get_workers().await.expect("workers failed");
    // Default state has no workers
    assert!(workers.is_empty());
}

#[tokio::test]
async fn test_list_models_shape() {
    let client = setup().await;
    let models: Vec<ModelInfo> = client.list_models().await.expect("list_models failed");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, 1);
}

#[tokio::test]
async fn test_get_model_shape() {
    let client = setup().await;
    let model: ModelInfo = client.get_model(1).await.expect("get_model failed");
    assert_eq!(model.id, 1);
}

#[tokio::test]
async fn test_register_model_returns_u64() {
    let client = setup().await;
    let model_id: u64 = client
        .register_model("test", "Qm...", 0.1)
        .await
        .expect("register_model failed");
    // Node defaults to 1 if no model_id provided in params
    assert!(model_id >= 1);
}

#[tokio::test]
async fn test_start_and_stop_training() {
    let client = setup().await;
    client
        .start_training(1, 10, 60)
        .await
        .expect("start_training failed");
    client.stop_training(1).await.expect("stop_training failed");
}

#[tokio::test]
async fn test_training_result_shape() {
    let client = setup().await;
    let result: TrainingResultData = client
        .get_training_result(0)
        .await
        .expect("training_result failed");
    assert!(result.round_id == 0);
}

#[tokio::test]
async fn test_generate_proof_shape() {
    let client = setup().await;
    let ack: GenerateProofAck = client
        .generate_proof(1, 0)
        .await
        .expect("generate_proof failed");
    assert!(ack.accepted);
    assert_eq!(ack.round_id, 0);
}

#[tokio::test]
async fn test_submit_proof_returns_hash() {
    let client = setup().await;
    let submission = ProofSubmission {
        proof: vec![0xde, 0xad, 0xbe, 0xef],
        public_inputs: vec!["0x01".into(), "0x02".into()],
        model_id: 1,
        round_id: 1,
        old_state_hash: ("0x00".into(), "0x00".into()),
        new_state_hash: ("0x01".into(), "0x01".into()),
        loss: "0x00".into(),
        error_bound: "0x00".into(),
        step_number: 1,
    };
    let hash: String = client.submit_proof(&submission).await.expect("submit_proof failed");
    assert!(hash.starts_with("0x"), "proof hash should start with 0x");
}

#[tokio::test]
async fn test_stake_and_unstake() {
    let client = setup().await;
    let s: String = client.stake(1, 1.0).await.expect("stake failed");
    assert!(!s.is_empty());
    let u: String = client.unstake(1).await.expect("unstake failed");
    assert!(!u.is_empty());
}

#[tokio::test]
async fn test_staking_info_shape() {
    let client = setup().await;
    let info: StakingInfo = client
        .get_staking_info(1)
        .await
        .expect("staking_info failed");
    assert!(info.slashing_events.is_empty());
}

#[tokio::test]
async fn test_training_progress_shape() {
    let client = setup().await;
    let progress: Vec<TrainingProgress> = client
        .get_training_progress(1, 0)
        .await
        .expect("training_progress failed");
    assert!(progress.is_empty()); // node doesn't track history
}

#[tokio::test]
async fn test_get_current_round_shape() {
    let client = setup().await;
    let round: RoundInfo = client
        .get_current_round(1)
        .await
        .expect("get_current_round failed");
    assert_eq!(round.model_id, 1);
}

#[tokio::test]
async fn test_get_round_shape() {
    let client = setup().await;
    let round: RoundInfo = client
        .get_round(1, 0)
        .await
        .expect("get_round failed");
    assert_eq!(round.round_id, 0);
    assert_eq!(round.model_id, 1);
}

#[tokio::test]
async fn test_verify_proof() {
    let client = setup().await;
    let submission = ProofSubmission {
        proof: vec![0xab],
        public_inputs: vec!["0x01".into()],
        model_id: 1,
        round_id: 1,
        old_state_hash: ("0x00".into(), "0x00".into()),
        new_state_hash: ("0x01".into(), "0x01".into()),
        loss: "0x00".into(),
        error_bound: "0x00".into(),
        step_number: 1,
    };
    let valid: bool = client
        .verify_proof(&submission)
        .await
        .expect("verify_proof failed");
    assert!(valid);
}

#[tokio::test]
async fn test_get_worker_shape() {
    let client = setup().await;
    // No workers in default state, so this should return an RPC error
    let result = client.get_worker("nonexistent").await;
    assert!(result.is_err(), "get_worker for nonexistent worker should fail");
}

#[tokio::test]
async fn test_get_self_worker_shape() {
    let client = setup().await;
    let worker: WorkerInfo = client
        .get_self_worker()
        .await
        .expect("get_self_worker failed");
    assert!(!worker.id.is_empty());
}

#[tokio::test]
async fn test_claim_rewards() {
    let client = setup().await;
    let result: String = client
        .claim_rewards(1)
        .await
        .expect("claim_rewards failed");
    assert!(!result.is_empty());
}

/// Comprehensive test: exercises connect() then every method sequentially.
#[tokio::test]
async fn test_full_rpc_roundtrip() {
    let client = setup().await;

    // 1. Connect
    client.connect().await.expect("connect");

    // 2. Status queries
    let _ts: TrainingStatus = client.get_training_status().await.expect("ts");
    let _ns: NetworkStatus = client.get_network_status().await.expect("ns");
    let _ps: ProofStatus = client.get_proof_status().await.expect("ps");
    let _ws: Vec<WorkerInfo> = client.get_workers().await.expect("ws");
    let _ms: Vec<ModelInfo> = client.list_models().await.expect("ms");

    // 3. Model registration
    let _mid: u64 = client.register_model("test", "Qm...", 0.1).await.expect("rm");

    // 4. Training lifecycle
    client.start_training(1, 5, 60).await.expect("start");
    let _tr: TrainingResultData = client.get_training_result(0).await.expect("tr");
    let _gp: GenerateProofAck = client.generate_proof(1, 0).await.expect("gp");
    assert!(_gp.accepted);

    // 5. Proof submission
    let sub = ProofSubmission {
        proof: vec![0xca, 0xfe],
        public_inputs: vec!["0x01".into()],
        model_id: 1,
        round_id: 0,
        old_state_hash: ("0x0".into(), "0x0".into()),
        new_state_hash: ("0x1".into(), "0x1".into()),
        loss: "0x0".into(),
        error_bound: "0x0".into(),
        step_number: 0,
    };
    let _hash: String = client.submit_proof(&sub).await.expect("sp");

    // 6. Stop training
    client.stop_training(1).await.expect("stop");

    // 7. Staking
    let _s: String = client.stake(1, 1.0).await.expect("stake");
    let _u: String = client.unstake(1).await.expect("unstake");
    let _si: StakingInfo = client.get_staking_info(1).await.expect("si");

    // 8. Rounds
    let _cr: RoundInfo = client.get_current_round(1).await.expect("cr");
    let _r: RoundInfo = client.get_round(1, 0).await.expect("r");

    // 9. Self worker
    let _sw: WorkerInfo = client.get_self_worker().await.expect("sw");

    // 10. Claim rewards
    let _cl: String = client.claim_rewards(1).await.expect("cl");
}
