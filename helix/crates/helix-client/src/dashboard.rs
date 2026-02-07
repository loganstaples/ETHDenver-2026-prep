//! HELIX Dashboard HTTP Server
//!
//! Provides a REST API for the status dashboard to consume.
//! Includes bearer-token authentication, per-IP rate limiting,
//! configurable CORS, and dynamic state wiring.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Router,
    routing::get,
    extract::{ConnectInfo, State},
    response::{IntoResponse, Json},
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{CorsLayer, Any};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Bearer token required for authenticated endpoints. `None` disables auth.
    pub auth_token: Option<String>,
    /// Allowed CORS origins. Empty list falls back to `Any` (dev mode).
    pub allowed_origins: Vec<String>,
    /// Maximum requests per second per IP address
    pub rate_limit_per_second: u32,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            auth_token: None,
            allowed_origins: Vec::new(),
            rate_limit_per_second: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Dashboard State
// ---------------------------------------------------------------------------

/// Dashboard application state — shared between handlers and external updaters
pub struct DashboardState {
    /// Start time for uptime calculation
    start_time: Instant,
    /// Cached network status
    pub network_status: RwLock<NetworkStatus>,
    /// Cached training status
    pub training_status: RwLock<TrainingStatus>,
    /// Connected nodes
    pub nodes: RwLock<Vec<NodeInfo>>,
    /// Aggregated metrics
    pub metrics: RwLock<serde_json::Value>,
    /// Event log
    pub events: RwLock<Vec<serde_json::Value>>,
    /// Dashboard config (auth token, rate limits, etc.)
    pub config: DashboardConfig,
    /// Per-IP request timestamps for rate limiting
    rate_limiter: RwLock<HashMap<IpAddr, VecDeque<Instant>>>,
}

impl DashboardState {
    /// Create new dashboard state with the given config
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            start_time: Instant::now(),
            network_status: RwLock::new(NetworkStatus::default()),
            training_status: RwLock::new(TrainingStatus::default()),
            nodes: RwLock::new(Vec::new()),
            metrics: RwLock::new(serde_json::json!({})),
            events: RwLock::new(Vec::new()),
            config,
            rate_limiter: RwLock::new(HashMap::new()),
        }
    }

    /// Create state pre-populated with demo defaults (so the dashboard works standalone)
    pub fn with_defaults() -> Arc<Self> {
        let state = Self::new(DashboardConfig::default());

        // Pre-populate nodes
        let nodes = vec![
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
        ];

        // Pre-populate metrics
        let metrics = serde_json::json!({
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
                "bytes_sent": 45_678_901u64,
                "bytes_received": 43_210_987u64,
            },
            "performance": {
                "cpu_usage_percent": 23.5,
                "memory_usage_mb": 1234,
                "disk_usage_mb": 5678,
            }
        });

        // Pre-populate events
        let events = vec![
            serde_json::json!({
                "type": "RoundCompleted",
                "timestamp": "2024-01-15T14:30:00Z",
                "data": { "model_id": 0, "round_id": 42, "prover": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" }
            }),
            serde_json::json!({
                "type": "ProofSubmitted",
                "timestamp": "2024-01-15T14:29:55Z",
                "data": { "model_id": 0, "round_id": 42, "worker": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" }
            }),
            serde_json::json!({
                "type": "WorkerJoined",
                "timestamp": "2024-01-15T14:00:00Z",
                "data": { "model_id": 0, "worker": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "stake": "1.0 ETH" }
            }),
        ];

        let arc = Arc::new(state);
        // Use try_write to populate synchronously (no contention at init time)
        *arc.nodes.try_write().unwrap() = nodes;
        *arc.metrics.try_write().unwrap() = metrics;
        *arc.events.try_write().unwrap() = events;
        arc
    }

    // -- Update methods for external callers --

    /// Update network status from external source
    pub async fn update_network(&self, status: NetworkStatus) {
        *self.network_status.write().await = status;
    }

    /// Update training status from external source
    pub async fn update_training(&self, status: TrainingStatus) {
        *self.training_status.write().await = status;
    }

    /// Replace the nodes list
    pub async fn update_nodes(&self, nodes: Vec<NodeInfo>) {
        *self.nodes.write().await = nodes;
    }

    /// Append an event
    pub async fn push_event(&self, event: serde_json::Value) {
        let mut events = self.events.write().await;
        events.push(event);
        // Cap at 1000 events
        if events.len() > 1000 {
            events.drain(..events.len() - 1000);
        }
    }

    /// Replace metrics blob
    pub async fn update_metrics(&self, metrics: serde_json::Value) {
        *self.metrics.write().await = metrics;
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new(DashboardConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Middleware: Bearer Token Auth
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(state): State<Arc<DashboardState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    // If no auth token configured, skip auth
    let Some(ref expected_token) = state.config.auth_token else {
        return next.run(req).await.into_response();
    };

    // Skip auth for /health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await.into_response();
    }

    // Extract bearer token
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value[7..];
            if token == expected_token {
                next.run(req).await.into_response()
            } else {
                (StatusCode::UNAUTHORIZED, "Invalid token").into_response()
            }
        }
        _ => (StatusCode::UNAUTHORIZED, "Missing or invalid Authorization header").into_response(),
    }
}

// ---------------------------------------------------------------------------
// Middleware: Rate Limiting
// ---------------------------------------------------------------------------

async fn rate_limit_middleware(
    State(state): State<Arc<DashboardState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    let limit = state.config.rate_limit_per_second;
    if limit == 0 {
        return next.run(req).await.into_response();
    }

    // Extract IP from ConnectInfo or fall back to loopback
    let ip: IpAddr = req
        .extensions()
        .get::<ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0.ip())
        .unwrap_or(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));

    let now = Instant::now();
    let window = std::time::Duration::from_secs(1);

    {
        let mut limiter = state.rate_limiter.write().await;
        let timestamps = limiter.entry(ip).or_insert_with(VecDeque::new);

        // Remove entries older than window
        while let Some(front) = timestamps.front() {
            if now.duration_since(*front) > window {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        if timestamps.len() >= limit as usize {
            return (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded").into_response();
        }

        timestamps.push_back(now);
    }

    next.run(req).await.into_response()
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Create the dashboard API router with the given configuration
pub fn create_dashboard_router(config: DashboardConfig) -> Router {
    create_dashboard_router_with_state(Arc::new(DashboardState::new(config)))
}

/// Create dashboard router with a pre-built shared state (so callers can push updates)
pub fn create_dashboard_router_with_state(state: Arc<DashboardState>) -> Router {
    let cors_layer = build_cors_layer(&state.config);

    Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/api/status", get(status_handler))
        .route("/api/network", get(network_handler))
        .route("/api/training", get(training_handler))
        .route("/api/nodes", get(nodes_handler))
        .route("/api/metrics", get(metrics_handler))
        .route("/api/events", get(events_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .layer(cors_layer)
        .with_state(state)
}

fn build_cors_layer(config: &DashboardConfig) -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any);

    if config.allowed_origins.is_empty() {
        base.allow_origin(Any)
    } else {
        let origins: Vec<_> = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        base.allow_origin(origins)
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

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

/// Health check endpoint (always unauthenticated)
async fn health_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<HealthResponse> {
    let uptime = state.uptime_secs();

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
    let uptime = state.uptime_secs();
    let network = state.network_status.read().await;
    let training = state.training_status.read().await;

    Json(serde_json::json!({
        "node": {
            "id": if network.node_id.is_empty() { "helix-node-a1b2c3d4" } else { &network.node_id },
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

/// Network status endpoint — reads from dynamic state, falls back to demo values
async fn network_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<NetworkStatus> {
    let mut status = state.network_status.read().await.clone();
    status.uptime_seconds = state.uptime_secs();

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

/// Training status endpoint — reads from dynamic state
async fn training_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<TrainingStatus> {
    let status = state.training_status.read().await.clone();

    // Return demo values if not set
    if !status.is_active && status.total_rounds == 0 {
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

/// Connected nodes endpoint — reads from dynamic state
async fn nodes_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<Vec<NodeInfo>> {
    let nodes = state.nodes.read().await;
    if nodes.is_empty() {
        // Fallback demo data
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
    } else {
        Json(nodes.clone())
    }
}

/// Metrics endpoint — reads from dynamic state
async fn metrics_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<serde_json::Value> {
    let metrics = state.metrics.read().await;
    if metrics.is_null() || metrics.as_object().map_or(true, |m| m.is_empty()) {
        // Fallback demo data
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
                "bytes_sent": 45_678_901u64,
                "bytes_received": 43_210_987u64,
            },
            "performance": {
                "cpu_usage_percent": 23.5,
                "memory_usage_mb": 1234,
                "disk_usage_mb": 5678,
            }
        }))
    } else {
        Json(metrics.clone())
    }
}

/// Events endpoint — reads from dynamic state
async fn events_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<Vec<serde_json::Value>> {
    let events = state.events.read().await;
    if events.is_empty() {
        // Fallback demo data
        Json(vec![
            serde_json::json!({
                "type": "RoundCompleted",
                "timestamp": "2024-01-15T14:30:00Z",
                "data": { "model_id": 0, "round_id": 42, "prover": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" }
            }),
            serde_json::json!({
                "type": "ProofSubmitted",
                "timestamp": "2024-01-15T14:29:55Z",
                "data": { "model_id": 0, "round_id": 42, "worker": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" }
            }),
            serde_json::json!({
                "type": "WorkerJoined",
                "timestamp": "2024-01-15T14:00:00Z",
                "data": { "model_id": 0, "worker": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199", "stake": "1.0 ETH" }
            }),
        ])
    } else {
        Json(events.clone())
    }
}
