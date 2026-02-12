//! Integration test: Client SDK ↔ Node RPC Integration
//!
//! Starts a real node RPC server in-process, creates a `HelixClient` that
//! connects via JSON-RPC, and exercises the full client lifecycle:
//! register model → start training → monitor progress → get results.
//!
//! This validates that the client SDK actually works with real running nodes,
//! not just mocks.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use helix_client::config::ConfigProfile;
use helix_client::{ModelArchitecture, SdkModelConfig, TrainingParams};
use helix_client::session::TrainingEvent;
use helix_client::HelixClient;
use helix_node::api::http::{OrchestratorSnapshot, RoundInfo as NodeRoundInfo, WorkerInfo as NodeWorkerInfo};
use helix_node::api::rpc::{create_default_rpc_state, start_rpc_server, RpcState};

/// Find a free TCP port by binding to port 0 and reading the assigned port.
async fn free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Boot the node RPC server and return (state_handle, rpc_url).
/// The state handle allows tests to mutate node state to simulate training progress.
async fn boot_node() -> (Arc<RpcState>, String) {
    let port = free_port().await;
    let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    let state = create_default_rpc_state("aggregator", &addr.to_string());

    // Spawn a mock orchestrator that listens for round triggers and
    // marks training as active (simulates what a real orchestrator does).
    let orch_state = state.clone();
    let mut round_rx = state.round_trigger_tx.subscribe();
    tokio::spawn(async move {
        while round_rx.recv().await.is_ok() {
            // When start_training triggers, mark the node as actively training
            let mut snap = orch_state.snapshot.write();
            if snap.current_round.is_none() {
                snap.current_round = Some(NodeRoundInfo {
                    round_id: 0,
                    phase: "initializing".to_string(),
                    gradients_received: 0,
                    workers_assigned: 0,
                    commitment_hash: None,
                });
            }
        }
    });

    // Keep a stop subscriber alive so stop_trigger_tx.send() doesn't fail
    let mut stop_rx = state.stop_trigger_tx.subscribe();
    tokio::spawn(async move {
        let _ = stop_rx.recv().await;
    });

    let server_state = state.clone();
    tokio::spawn(async move {
        start_rpc_server(addr, server_state).await.unwrap();
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let url = format!("http://127.0.0.1:{}/rpc", port);
    (state, url)
}

/// Simulate training progress by updating the node's snapshot.
fn advance_node_state(state: &RpcState, round_id: u64, phase: &str, completed_rounds: u64) {
    let mut snap = state.snapshot.write();
    *snap = OrchestratorSnapshot {
        worker_count: 2,
        available_workers: 1,
        computing_workers: 1,
        completed_rounds,
        current_round: Some(NodeRoundInfo {
            round_id,
            phase: phase.to_string(),
            gradients_received: 2,
            workers_assigned: 2,
            commitment_hash: Some(format!("0x{}", "ab".repeat(32))),
        }),
        workers: vec![
            NodeWorkerInfo {
                id: "worker-1".to_string(),
                status: "training".to_string(),
                rounds_completed: completed_rounds,
            },
            NodeWorkerInfo {
                id: "worker-2".to_string(),
                status: "training".to_string(),
                rounds_completed: completed_rounds,
            },
        ],
    };
}

/// Clear the current round to signal training is done.
fn finish_training(state: &RpcState, completed_rounds: u64) {
    let mut snap = state.snapshot.write();
    snap.current_round = None;
    snap.completed_rounds = completed_rounds;
    snap.computing_workers = 0;
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn test_client_connects_to_real_node() {
    let (_state, url) = boot_node().await;

    // Use connect_to() which creates a client and connects in one call
    let client = HelixClient::connect_to(&url)
        .await
        .expect("Should connect to real node");

    assert!(client.is_connected());
}

#[tokio::test]
async fn test_client_connect_to_convenience() {
    let (_state, url) = boot_node().await;

    let client = HelixClient::connect_to(&url)
        .await
        .expect("connect_to should succeed");

    assert!(client.is_connected());

    // Health check should work
    let health = client.health().await.expect("health check failed");
    assert!(health.healthy);
}

#[tokio::test]
async fn test_client_register_model() {
    let (state, url) = boot_node().await;
    // Pre-set model_id so the node returns it
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let model_config = SdkModelConfig::new(
        "test-mnist",
        ModelArchitecture::new(784, 128, 10),
    );

    let handle = client.register_model(model_config).await.expect("register_model failed");
    assert!(handle.model_id >= 1);
    assert_eq!(handle.name, "test-mnist");
}

#[tokio::test]
async fn test_client_node_status() {
    let (state, url) = boot_node().await;
    // Set up some workers in the snapshot
    {
        let mut snap = state.snapshot.write();
        snap.worker_count = 3;
        snap.available_workers = 2;
    }

    let client = HelixClient::connect_to(&url).await.unwrap();
    let status = client.node_status().await.expect("node_status failed");

    assert_eq!(status.peer_count, 3);
    assert_eq!(status.active_workers, 2);
    assert_eq!(status.chain_id, 31337);
}

#[tokio::test]
async fn test_client_training_status() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();
    let status = client.training_status().await.expect("training_status failed");

    // No active training
    assert!(!status.active);
    assert_eq!(status.model_id, 1);
}

#[tokio::test]
async fn test_client_start_training_and_poll() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let params = TrainingParams {
        rounds: 3,
        round_duration_secs: 60,
        poll_interval_ms: 50, // Fast polling for test
        ..TrainingParams::default()
    };

    let mut session = client
        .start_training(1, params)
        .await
        .expect("start_training failed");

    // Simulate the node processing round 1
    tokio::time::sleep(Duration::from_millis(30)).await;
    advance_node_state(&state, 1, "forward", 0);

    // Collect events until we see RoundStarted
    let mut saw_round_started = false;
    let timeout = tokio::time::sleep(Duration::from_secs(3));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            event = session.next_event() => {
                match event {
                    Some(TrainingEvent::RoundStarted { round, total }) => {
                        assert_eq!(round, 1);
                        assert!(total >= 1);
                        saw_round_started = true;
                        break;
                    }
                    Some(TrainingEvent::Error { message }) => {
                        // Transient errors are ok while we're setting up state
                        eprintln!("Transient error: {}", message);
                        continue;
                    }
                    None => break,
                    _ => continue,
                }
            }
            _ = &mut timeout => {
                break;
            }
        }
    }

    assert!(saw_round_started, "Should have received RoundStarted event");
    session.cancel();
}

#[tokio::test]
async fn test_client_full_training_lifecycle() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let params = TrainingParams {
        rounds: 2,
        round_duration_secs: 60,
        poll_interval_ms: 50,
        ..TrainingParams::default()
    };

    let mut session = client
        .start_training(1, params)
        .await
        .expect("start_training failed");

    // Spawn a task to simulate training progress on the node
    let sim_state = state.clone();
    let sim_task = tokio::spawn(async move {
        // Round 1: forward → proof → complete
        tokio::time::sleep(Duration::from_millis(80)).await;
        advance_node_state(&sim_state, 1, "forward", 0);

        tokio::time::sleep(Duration::from_millis(100)).await;
        advance_node_state(&sim_state, 1, "proof_generation", 0);

        tokio::time::sleep(Duration::from_millis(100)).await;
        advance_node_state(&sim_state, 2, "forward", 1);

        // Round 2: forward → done
        tokio::time::sleep(Duration::from_millis(100)).await;
        finish_training(&sim_state, 2);
    });

    // Collect all events
    let mut events = Vec::new();
    let timeout = tokio::time::sleep(Duration::from_secs(5));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            event = session.next_event() => {
                match event {
                    Some(TrainingEvent::TrainingComplete { .. }) => {
                        events.push(event.unwrap());
                        break;
                    }
                    Some(e) => {
                        events.push(e);
                    }
                    None => break,
                }
            }
            _ = &mut timeout => {
                break;
            }
        }
    }

    sim_task.await.unwrap();

    // Should have seen: RoundStarted, StepCompleted, ProofGenerated, RoundStarted (round 2),
    // RoundCompleted (round 1), StepCompleted, TrainingComplete
    let round_started_count = events.iter().filter(|e| matches!(e, TrainingEvent::RoundStarted { .. })).count();
    let training_complete = events.iter().any(|e| matches!(e, TrainingEvent::TrainingComplete { .. }));

    assert!(round_started_count >= 1, "Should have at least 1 RoundStarted, got {}", round_started_count);
    assert!(training_complete, "Should have received TrainingComplete");
}

#[tokio::test]
async fn test_client_wait_for_result() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let params = TrainingParams {
        rounds: 1,
        round_duration_secs: 60,
        poll_interval_ms: 50,
        ..TrainingParams::default()
    };

    let mut session = client
        .start_training(1, params)
        .await
        .expect("start_training failed");

    // Simulate quick training
    let sim_state = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        advance_node_state(&sim_state, 1, "forward", 0);
        tokio::time::sleep(Duration::from_millis(100)).await;
        finish_training(&sim_state, 1);
    });

    // wait() blocks until training completes
    let result = tokio::time::timeout(Duration::from_secs(5), session.wait())
        .await
        .expect("wait() timed out")
        .expect("wait() returned error");

    assert!(result.success);
}

#[tokio::test]
async fn test_client_cancel_training() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let params = TrainingParams {
        rounds: 100, // Long training
        round_duration_secs: 60,
        poll_interval_ms: 50,
        ..TrainingParams::default()
    };

    let mut session = client
        .start_training(1, params)
        .await
        .expect("start_training failed");

    // Cancel after a brief wait
    tokio::time::sleep(Duration::from_millis(100)).await;
    session.cancel();

    // Should complete quickly with success=false
    let result = tokio::time::timeout(Duration::from_secs(2), session.wait())
        .await
        .expect("wait() timed out after cancel")
        .expect("wait() returned error");

    assert!(!result.success, "Cancelled training should not report success");
}

#[tokio::test]
async fn test_client_get_training_result() {
    let (_state, url) = boot_node().await;

    let client = HelixClient::connect_to(&url).await.unwrap();

    let result = client
        .get_training_result(0)
        .await
        .expect("get_training_result failed");

    assert_eq!(result.round_id, 0);
}

#[tokio::test]
async fn test_client_stake_via_node() {
    let (_state, url) = boot_node().await;

    let client = HelixClient::connect_to(&url).await.unwrap();

    let tx_hash = client.stake(1, 1.0).await.expect("stake failed");
    assert!(!tx_hash.is_empty());

    let tx_hash = client.unstake(1).await.expect("unstake failed");
    assert!(!tx_hash.is_empty());
}

#[tokio::test]
async fn test_client_disconnected_errors() {
    // A client that hasn't connected should error on RPC calls
    let client = HelixClient::from_profile(ConfigProfile::Local).unwrap();
    assert!(!client.is_connected());

    let result = client.health().await;
    assert!(result.is_err(), "Disconnected client should error on health()");

    let result = client.node_status().await;
    assert!(result.is_err(), "Disconnected client should error on node_status()");
}

#[tokio::test]
async fn test_default_rpc_client_is_not_mock() {
    // Verify the Default impl returns disconnected, not mock
    let rpc = helix_client::rpc::client::UnifiedRpcClient::default();
    assert!(!rpc.is_mock(), "Default UnifiedRpcClient should NOT be mock");
    assert!(!rpc.is_connected(), "Default UnifiedRpcClient should be disconnected");
}

#[tokio::test]
async fn test_client_progress_during_training() {
    let (state, url) = boot_node().await;
    *state.model_id.write() = Some(1);

    let client = HelixClient::connect_to(&url).await.unwrap();

    let params = TrainingParams {
        rounds: 5, // More rounds so training won't auto-complete
        round_duration_secs: 60,
        poll_interval_ms: 50,
        ..TrainingParams::default()
    };

    let session = client
        .start_training(1, params)
        .await
        .expect("start_training failed");

    // Advance to round 1
    tokio::time::sleep(Duration::from_millis(100)).await;
    advance_node_state(&state, 1, "forward", 0);

    // Give the poller 3 poll intervals to pick up the change
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check progress snapshot
    let progress = session.progress().await;
    assert_eq!(progress.current_round, 1, "should have polled round 1");
}
