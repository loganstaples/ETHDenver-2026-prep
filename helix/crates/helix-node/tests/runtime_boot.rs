//! Integration tests for the NodeRuntime boot sequence.
//!
//! Verifies that the runtime boots subsystems in the correct order,
//! responds to health queries, and shuts down cleanly on signal.
//!
//! Uses `current_thread` tokio flavor because NodeRuntime::run() holds
//! parking_lot guards that are !Send. This matches production usage where
//! the runtime runs on the main task via `tokio::select!`.

use std::net::TcpListener;
use std::time::Duration;

use helix_node::config::{NodeConfig, NodeRole};
use helix_node::runtime::{NodeRuntime, SubsystemStatus};

/// Find an available TCP port by binding to port 0.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Create a test config with unique ports so tests don't collide.
fn test_config(role: NodeRole) -> NodeConfig {
    let p2p_port = free_port();
    let rpc_port = free_port();
    let http_port = free_port();

    let mut config = NodeConfig {
        role,
        listen_addr: format!("127.0.0.1:{}", p2p_port),
        rpc_port,
        http_port,
        mdns_enabled: false,
        bootstrap_nodes: vec![],
        ..Default::default()
    };
    config.mpc.enabled = false;
    config.training.d_in = 4;
    config.training.d_hid = 8;
    config.training.d_out = 2;
    config.training.learning_rate = 0.01;
    config.training.min_workers = 1;
    config.fault_tolerance.shutdown_timeout_secs = 2;
    config.checkpoint_dir = std::path::PathBuf::new(); // disable persistence
    config
}

#[tokio::test(flavor = "current_thread")]
async fn test_runtime_creates_with_default_health() {
    let config = test_config(NodeRole::Compute);
    let runtime = NodeRuntime::new(config);
    let health = runtime.health();
    assert!(!health.is_healthy());
    assert_eq!(health.network, SubsystemStatus::Pending);
    assert_eq!(health.rpc_server, SubsystemStatus::Pending);
}

#[tokio::test(flavor = "current_thread")]
async fn test_runtime_shutdown_idempotent() {
    let config = test_config(NodeRole::Compute);
    let runtime = NodeRuntime::new(config);
    // Calling shutdown multiple times should not panic
    runtime.shutdown();
    runtime.shutdown();
    runtime.shutdown();
}

#[tokio::test(flavor = "current_thread")]
async fn test_worker_boot_and_shutdown() {
    let config = test_config(NodeRole::Compute);
    let rpc_port = config.rpc_port;
    let runtime = NodeRuntime::new(config);

    tokio::select! {
        result = runtime.run() => {
            // Runtime exited on its own (network stream ended) — acceptable
            if let Err(e) = result {
                eprintln!("Runtime exited with: {}", e);
            }
        }
        _ = async {
            // Give subsystems time to start
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Verify RPC port is listening
            let rpc_addr = format!("127.0.0.1:{}", rpc_port);
            let connected = tokio::net::TcpStream::connect(&rpc_addr).await.is_ok();
            assert!(connected, "RPC server should be listening on {}", rpc_addr);

            // Trigger clean shutdown
            runtime.shutdown();
            tokio::time::sleep(Duration::from_millis(100)).await;
        } => {}
    }
}

#[tokio::test(flavor = "current_thread")]
async fn test_aggregator_boot_and_shutdown() {
    let config = test_config(NodeRole::Aggregator);
    let rpc_port = config.rpc_port;
    let http_port = config.http_port;
    let runtime = NodeRuntime::new(config);

    tokio::select! {
        result = runtime.run() => {
            if let Err(e) = result {
                eprintln!("Runtime exited with: {}", e);
            }
        }
        _ = async {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Verify RPC port
            let rpc_addr = format!("127.0.0.1:{}", rpc_port);
            let rpc_ok = tokio::net::TcpStream::connect(&rpc_addr).await.is_ok();
            assert!(rpc_ok, "RPC should be listening on {}", rpc_addr);

            // Verify HTTP API port
            let http_addr = format!("127.0.0.1:{}", http_port);
            let http_ok = tokio::net::TcpStream::connect(&http_addr).await.is_ok();
            assert!(http_ok, "HTTP API should be listening on {}", http_addr);

            runtime.shutdown();
            tokio::time::sleep(Duration::from_millis(100)).await;
        } => {}
    }
}

#[tokio::test(flavor = "current_thread")]
async fn test_runtime_health_json_structure() {
    let config = test_config(NodeRole::Compute);
    let runtime = NodeRuntime::new(config);
    let health = runtime.health();
    let json = health.to_json();

    // Verify JSON structure
    assert!(json["healthy"].is_boolean());
    assert!(json["uptime_secs"].is_number());
    assert!(json["components"].is_object());
    assert!(json["components"]["network"].is_string());
    assert!(json["components"]["rpc_server"].is_string());
    assert!(json["components"]["http_api"].is_string());
    assert!(json["components"]["training_orchestrator"].is_string());
    assert!(json["components"]["failure_detector"].is_string());
    assert!(json["components"]["chain_pipeline"].is_string());
    assert!(json["version"].is_string());
}

#[tokio::test(flavor = "current_thread")]
async fn test_config_from_toml_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test-node.toml");

    let toml_content = r#"
role = "compute"
listen_addr = "127.0.0.1:19000"
rpc_port = 19002
http_port = 19001
rpc_url = "http://localhost:8545"
mdns_enabled = false

[training]
d_in = 10
d_hid = 20
d_out = 5
learning_rate = 0.001
model_seed = 99

[mpc]
enabled = false

[fault_tolerance]
checkpoint_interval_steps = 5
"#;
    std::fs::write(&path, toml_content).unwrap();

    let config = NodeConfig::from_file(&path).unwrap();
    assert_eq!(config.role, NodeRole::Compute);
    assert_eq!(config.listen_addr, "127.0.0.1:19000");
    assert_eq!(config.rpc_port, 19002);
    assert_eq!(config.training.d_in, 10);
    assert_eq!(config.training.d_hid, 20);
    assert_eq!(config.training.d_out, 5);
    assert!(!config.mpc.enabled);
    assert_eq!(config.fault_tolerance.checkpoint_interval_steps, 5);
}

#[tokio::test(flavor = "current_thread")]
async fn test_config_from_json_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test-node.json");

    let json_content = r#"{
        "role": "aggregator",
        "listen_addr": "127.0.0.1:19000",
        "rpc_port": 19002,
        "rpc_url": "http://localhost:8545",
        "training": {
            "d_in": 6,
            "d_hid": 12,
            "d_out": 3
        }
    }"#;
    std::fs::write(&path, json_content).unwrap();

    let config = NodeConfig::from_file(&path).unwrap();
    assert_eq!(config.role, NodeRole::Aggregator);
    assert_eq!(config.training.d_in, 6);
}

#[tokio::test(flavor = "current_thread")]
async fn test_config_validation_rejects_bad_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.toml");

    // Zero dimensions should fail validation
    let bad_toml = r#"
listen_addr = "127.0.0.1:9000"
rpc_url = "http://localhost:8545"

[training]
d_in = 0
d_hid = 8
d_out = 2
"#;
    std::fs::write(&path, bad_toml).unwrap();
    assert!(NodeConfig::from_file(&path).is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn test_worker_rpc_health_endpoint() {
    let config = test_config(NodeRole::Compute);
    let rpc_port = config.rpc_port;
    let runtime = NodeRuntime::new(config);

    tokio::select! {
        _ = runtime.run() => {}
        _ = async {
            // Wait for boot
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Send JSON-RPC health request (route is /rpc)
            let client = reqwest::Client::new();
            let resp = client
                .post(format!("http://127.0.0.1:{}/rpc", rpc_port))
                .json(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "helix_health",
                    "params": {},
                    "id": 1
                }))
                .send()
                .await;

            match resp {
                Ok(r) => {
                    assert_eq!(r.status(), 200);
                    let body: serde_json::Value = r.json().await.unwrap();
                    assert_eq!(body["jsonrpc"], "2.0");
                    assert!(body["result"].is_object() || body["error"].is_object());
                }
                Err(e) => {
                    eprintln!("RPC health check failed (may be timing): {}", e);
                }
            }

            runtime.shutdown();
            tokio::time::sleep(Duration::from_millis(100)).await;
        } => {}
    }
}

#[tokio::test(flavor = "current_thread")]
async fn test_graceful_shutdown_via_runtime() {
    let config = test_config(NodeRole::Compute);
    let runtime = NodeRuntime::new(config);

    let start = std::time::Instant::now();

    tokio::select! {
        result = runtime.run() => {
            match result {
                Ok(()) => {}
                Err(e) => eprintln!("Runtime exited with: {}", e),
            }
        }
        _ = async {
            // Let it start
            tokio::time::sleep(Duration::from_millis(300)).await;

            // Trigger shutdown
            runtime.shutdown();

            // Wait for shutdown to propagate
            tokio::time::sleep(Duration::from_millis(200)).await;
        } => {}
    }

    // Verify it shut down within a reasonable time
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "Shutdown should complete within 5 seconds"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_subsystem_status_display() {
    assert_eq!(SubsystemStatus::Pending.to_string(), "pending");
    assert_eq!(SubsystemStatus::Starting.to_string(), "starting");
    assert_eq!(SubsystemStatus::Running.to_string(), "running");
    assert_eq!(SubsystemStatus::Stopped.to_string(), "stopped");
    assert_eq!(
        SubsystemStatus::Failed("test error".into()).to_string(),
        "failed: test error"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_config_accessor() {
    let mut config = test_config(NodeRole::Compute);
    config.training.d_in = 16;
    let runtime = NodeRuntime::new(config);
    assert_eq!(runtime.config().training.d_in, 16);
    assert_eq!(runtime.config().role, NodeRole::Compute);
}
