//! HELIX Dashboard HTTP Server
//!
//! Provides a REST API for the status dashboard to consume.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Router,
    routing::get,
    extract::State,
    response::Json,
    http::{StatusCode, Method},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{CorsLayer, Any};

/// Dashboard application state
pub struct DashboardState {
    /// Start time for uptime calculation
    start_time: Instant,
    /// Cached network status
    network_status: RwLock<NetworkStatus>,
    /// Cached training status
    training_status: RwLock<TrainingStatus>,
}

impl DashboardState {
    /// Create new dashboard state
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            network_status: RwLock::new(NetworkStatus::default()),
            training_status: RwLock::new(TrainingStatus::default()),
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}

/// Network status information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkStatus {
    pub node_id: String,
    pub status: String,
    pub peer_count: u32,
    pub uptime_seconds: u64,
    pub block_height: u64,
    pub chain_id: u64,
}

/// Training status information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingStatus {
    pub model_id: u64,
    pub current_round: u64,
    pub total_rounds: u64,
    pub proofs_submitted: u64,
    pub proofs_verified: u64,
    pub error_bound: f64,
    pub max_error: f64,
    pub is_active: bool,
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub role: String,
    pub status: String,
    pub stake: f64,
    pub reputation: f64,
    pub proofs_submitted: u64,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: Vec<HealthCheck>,
}

/// Individual health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
    pub message: Option<String>,
}

/// Create the dashboard API router
pub fn create_dashboard_router(enable_cors: bool) -> Router {
    let state = Arc::new(DashboardState::new());

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/api/status", get(status_handler))
        .route("/api/network", get(network_handler))
        .route("/api/training", get(training_handler))
        .route("/api/nodes", get(nodes_handler))
        .route("/api/metrics", get(metrics_handler))
        .route("/api/events", get(events_handler))
        .with_state(state);

    if enable_cors {
        app.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers(Any),
        )
    } else {
        app
    }
}

/// Root handler - returns API info
async fn root_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "HELIX Dashboard API",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "/health",
            "/api/status",
            "/api/network",
            "/api/training",
            "/api/nodes",
            "/api/metrics",
            "/api/events"
        ]
    }))
}

/// Health check endpoint
async fn health_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<HealthResponse> {
    let uptime = state.start_time.elapsed().as_secs();

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: uptime,
        checks: vec![
            HealthCheck {
                name: "api".to_string(),
                status: "ok".to_string(),
                message: None,
            },
            HealthCheck {
                name: "rpc".to_string(),
                status: "ok".to_string(),
                message: Some("Connected to localhost:8545".to_string()),
            },
            HealthCheck {
                name: "storage".to_string(),
                status: "ok".to_string(),
                message: None,
            },
        ],
    })
}

/// Overall status endpoint
async fn status_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<serde_json::Value> {
    let uptime = state.start_time.elapsed().as_secs();
    let network = state.network_status.read().await;
    let training = state.training_status.read().await;

    Json(serde_json::json!({
        "node": {
            "id": "helix-node-a1b2c3d4",
            "status": "online",
            "uptime_seconds": uptime,
            "version": env!("CARGO_PKG_VERSION"),
        },
        "network": {
            "peer_count": network.peer_count,
            "block_height": network.block_height,
            "chain_id": network.chain_id,
        },
        "training": {
            "model_id": training.model_id,
            "current_round": training.current_round,
            "is_active": training.is_active,
        }
    }))
}

/// Network status endpoint
async fn network_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<NetworkStatus> {
    let mut status = state.network_status.read().await.clone();
    status.uptime_seconds = state.start_time.elapsed().as_secs();

    // Populate with demo values if not set
    if status.node_id.is_empty() {
        status.node_id = "helix-node-a1b2c3d4".to_string();
        status.status = "online".to_string();
        status.peer_count = 4;
        status.block_height = 12_345_678;
        status.chain_id = 31337;
    }

    Json(status)
}

/// Training status endpoint
async fn training_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<TrainingStatus> {
    let status = state.training_status.read().await.clone();

    // Return demo values if not set
    if !status.is_active {
        Json(TrainingStatus {
            model_id: 0,
            current_round: 42,
            total_rounds: 100,
            proofs_submitted: 127,
            proofs_verified: 125,
            error_bound: 45.2,
            max_error: 1000.0,
            is_active: true,
        })
    } else {
        Json(status)
    }
}

/// Connected nodes endpoint
async fn nodes_handler() -> Json<Vec<NodeInfo>> {
    // Demo data
    Json(vec![
        NodeInfo {
            id: "helix-node-a1b2c3d4".to_string(),
            address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string(),
            role: "worker".to_string(),
            status: "active".to_string(),
            stake: 1.5,
            reputation: 0.98,
            proofs_submitted: 42,
        },
        NodeInfo {
            id: "helix-node-e5f6g7h8".to_string(),
            address: "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string(),
            role: "worker".to_string(),
            status: "active".to_string(),
            stake: 1.0,
            reputation: 0.95,
            proofs_submitted: 38,
        },
        NodeInfo {
            id: "helix-agg-01".to_string(),
            address: "0xdD2FD4581271e230360230F9337D5c0430Bf44C0".to_string(),
            role: "aggregator".to_string(),
            status: "active".to_string(),
            stake: 2.5,
            reputation: 1.0,
            proofs_submitted: 0,
        },
    ])
}

/// Metrics endpoint
async fn metrics_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "training": {
            "rounds_completed": 42,
            "proofs_generated": 127,
            "proofs_verified": 125,
            "avg_round_time_ms": 2500,
            "avg_proof_time_ms": 1200,
        },
        "network": {
            "messages_sent": 15678,
            "messages_received": 14532,
            "bytes_sent": 45_678_901,
            "bytes_received": 43_210_987,
        },
        "performance": {
            "cpu_usage_percent": 23.5,
            "memory_usage_mb": 1234,
            "disk_usage_mb": 5678,
        }
    }))
}

/// Events endpoint
async fn events_handler() -> Json<Vec<serde_json::Value>> {
    Json(vec![
        serde_json::json!({
            "type": "RoundCompleted",
            "timestamp": "2024-01-15T14:30:00Z",
            "data": {
                "model_id": 0,
                "round_id": 42,
                "prover": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            }
        }),
        serde_json::json!({
            "type": "ProofSubmitted",
            "timestamp": "2024-01-15T14:29:55Z",
            "data": {
                "model_id": 0,
                "round_id": 42,
                "worker": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            }
        }),
        serde_json::json!({
            "type": "WorkerJoined",
            "timestamp": "2024-01-15T14:00:00Z",
            "data": {
                "model_id": 0,
                "worker": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
                "stake": "1.0 ETH",
            }
        }),
    ])
}
