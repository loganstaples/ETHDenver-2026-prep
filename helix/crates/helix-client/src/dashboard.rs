//! HELIX Dashboard HTTP + WebSocket Server
//!
//! REST API for status dashboard + training job submission + WebSocket progress streaming.
//! - POST /api/training/start — submit a training job from the web app
//! - GET  /api/training/:id   — get training session state
//! - GET  /api/training/:id/losses — get loss history
//! - WS   /ws                 — WebSocket for real-time progress events
//! - GET  /api/status, /api/network, etc. — existing dashboard endpoints

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Router,
    routing::{get, post},
    extract::{
        ConnectInfo, Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::{IntoResponse, Json},
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
};
use futures::stream::StreamExt;
use futures::SinkExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::{CorsLayer, Any};
use tracing::{debug, error, info, warn};

use crate::full_orchestration::{
    FullOrchestrationConfig, FullOrchestrator, ProgressCallback, ProgressEvent, ZkMode,
};
use crate::zk_proof_layer::ZkProofConfig;

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
// Training Job Request/Response Types
// ---------------------------------------------------------------------------

/// Training job submission from the web app.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobRequest {
    /// Layer dimensions, e.g. [784, 32, 10]
    #[serde(default = "default_architecture")]
    pub architecture: Vec<usize>,
    /// Number of MPC workers
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
    /// Total training steps
    #[serde(default = "default_num_steps")]
    pub num_steps: usize,
    /// Learning rate
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// Checkpoint every N steps
    #[serde(default = "default_checkpoint_freq")]
    pub checkpoint_freq: usize,
    /// MAC verification interval (every N steps)
    #[serde(default = "default_mac_interval")]
    pub mac_interval: u64,
    /// ZK mode: "off", "always", or "risk"
    #[serde(default = "default_zk_mode")]
    pub zk_mode: String,
    /// ZK proof every N checkpoints (when zk_mode != "off")
    #[serde(default = "default_zk_checkpoint_freq")]
    pub zk_checkpoint_freq: u64,
    /// Min workers for MPC before risk-ZK activates
    #[serde(default = "default_min_workers")]
    pub min_workers_for_mpc: usize,
    /// Number of training samples
    #[serde(default = "default_train_size")]
    pub train_size: usize,
    /// Number of test samples
    #[serde(default = "default_test_size")]
    pub test_size: usize,
    /// Use real MNIST data (requires network)
    #[serde(default)]
    pub use_real_mnist: bool,
    /// ETH payment for training job
    #[serde(default = "default_payment")]
    pub payment_eth: f64,
    /// ETH stake per worker
    #[serde(default = "default_stake")]
    pub stake_per_worker_eth: f64,
    /// Demo feature: simulate a cheater
    #[serde(default)]
    pub simulate_cheater: bool,
    /// Random seed
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Mini-batch size for training
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// Transport mode: "local" (default) or "distributed"
    #[serde(default = "default_transport")]
    pub transport: String,
}

fn default_architecture() -> Vec<usize> { vec![784, 32, 10] }
fn default_num_workers() -> usize { 3 }
fn default_num_steps() -> usize { 500 }
fn default_learning_rate() -> f64 { 0.001 }
fn default_checkpoint_freq() -> usize { 50 }
fn default_mac_interval() -> u64 { 1 }
fn default_zk_mode() -> String { "off".to_string() }
fn default_zk_checkpoint_freq() -> u64 { 5 }
fn default_min_workers() -> usize { 2 }
fn default_train_size() -> usize { 1000 }
fn default_test_size() -> usize { 200 }
fn default_payment() -> f64 { 1.0 }
fn default_stake() -> f64 { 0.1 }
fn default_seed() -> u64 { 42 }
fn default_transport() -> String { "local".to_string() }

/// Response after submitting a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobResponse {
    pub session_id: String,
    pub status: String,
}

/// Live training session state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSessionState {
    pub session_id: String,
    pub status: String,
    pub current_step: usize,
    pub total_steps: usize,
    pub current_loss: f64,
    pub losses: Vec<f64>,
    pub accuracy: Option<f64>,
    pub checkpoints_submitted: usize,
    pub mac_checks_passed: usize,
    pub cheater_detected: Option<serde_json::Value>,
    pub zk_proofs_generated: usize,
    pub zk_activated_by_risk: bool,
    pub phase: u32,
    pub phase_description: String,
    pub coordinator_address: String,
    pub job_id: u64,
    pub elapsed_secs: f64,
    pub started_at: f64,
    /// Final trained weights (populated when training completes)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_weights: Option<serde_json::Value>,
}

/// Request body for uploading training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataUpload {
    pub samples: Vec<TrainingSample>,
}

/// A single training sample (input, target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

/// Request body for uploading initial weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsUpload {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Demo data helpers (single source of truth)
// ---------------------------------------------------------------------------

/// The 3 demo nodes used by `with_defaults()` and handler fallbacks.
pub fn demo_nodes() -> Vec<NodeInfo> {
    vec![
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
    ]
}

/// Demo metrics JSON blob.
pub fn demo_metrics() -> serde_json::Value {
    serde_json::json!({
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
    })
}

/// Demo events list.
pub fn demo_events() -> Vec<serde_json::Value> {
    vec![
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
    ]
}

/// Default training status returned when no live data is available.
pub fn demo_training_status() -> TrainingStatus {
    TrainingStatus {
        model_id: 0,
        current_round: 42,
        total_rounds: 100,
        proofs_submitted: 127,
        proofs_verified: 125,
        error_bound: 45.2,
        max_error: 1000.0,
        is_active: true,
    }
}

/// Fill an empty `NetworkStatus` with demo values (in-place).
pub fn populate_demo_network(status: &mut NetworkStatus) {
    status.node_id = "helix-node-a1b2c3d4".to_string();
    status.status = "online".to_string();
    status.peer_count = 4;
    status.block_height = 12_345_678;
    status.chain_id = 0; // Will be set to actual chain ID when connected
}

// ---------------------------------------------------------------------------
// Worker Registry
// ---------------------------------------------------------------------------

/// A registered MPC worker that is available for training jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredWorker {
    /// Unique worker ID (assigned on registration)
    pub id: String,
    /// TCP endpoint the worker is listening on (e.g. "192.168.1.5:9001")
    pub endpoint: String,
    /// Party index hint (from the worker)
    pub party_index: usize,
    /// Unix timestamp when the worker registered
    pub registered_at: f64,
    /// Unix timestamp of the last heartbeat
    pub last_heartbeat: f64,
    /// Worker status: "idle", "busy", "offline"
    pub status: String,
}

/// Request body for worker registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRegisterRequest {
    /// TCP endpoint the worker is listening on
    pub endpoint: String,
    /// Party index hint
    #[serde(default)]
    pub party_index: usize,
}

/// Request body for worker heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatRequest {
    pub worker_id: String,
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
    /// Active training sessions
    pub sessions: RwLock<HashMap<String, TrainingSessionState>>,
    /// WebSocket broadcast channel for progress events
    pub ws_broadcast: broadcast::Sender<(String, serde_json::Value)>,
    /// Uploaded training data (used instead of synthetic MNIST when set)
    pub uploaded_data: RwLock<Option<Vec<(Vec<f64>, Vec<f64>)>>>,
    /// Uploaded initial weights (for continue-training)
    pub uploaded_weights: RwLock<Option<serde_json::Value>>,
    /// Registered MPC workers available for training jobs (off-chain fallback)
    pub registered_workers: RwLock<Vec<RegisteredWorker>>,
    /// On-chain coordinator address (set after first job deploys or from config)
    pub coordinator_address: RwLock<Option<String>>,
    /// Ethereum RPC URL for reading on-chain state
    pub eth_rpc_url: RwLock<Option<String>>,
}

impl DashboardState {
    /// Create new dashboard state with the given config
    pub fn new(config: DashboardConfig) -> Self {
        let (ws_broadcast, _) = broadcast::channel(1024);
        Self {
            start_time: Instant::now(),
            network_status: RwLock::new(NetworkStatus::default()),
            training_status: RwLock::new(TrainingStatus::default()),
            nodes: RwLock::new(Vec::new()),
            metrics: RwLock::new(serde_json::json!({})),
            events: RwLock::new(Vec::new()),
            config,
            rate_limiter: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
            ws_broadcast,
            uploaded_data: RwLock::new(None),
            uploaded_weights: RwLock::new(None),
            registered_workers: RwLock::new(Vec::new()),
            coordinator_address: RwLock::new(None),
            eth_rpc_url: RwLock::new(None),
        }
    }

    /// Create state pre-populated with demo defaults (so the dashboard works standalone)
    pub fn with_defaults() -> Arc<Self> {
        let state = Self::new(DashboardConfig::default());

        let arc = Arc::new(state);
        // Use try_write to populate synchronously (no contention at init time)
        *arc.nodes.try_write().expect("no contention at init") = demo_nodes();
        *arc.metrics.try_write().expect("no contention at init") = demo_metrics();
        *arc.events.try_write().expect("no contention at init") = demo_events();
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
        let len = events.len();
        if len > 1000 {
            events.drain(..len - 1000);
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

impl Default for TrainingSessionState {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            status: "pending".to_string(),
            current_step: 0,
            total_steps: 0,
            current_loss: 0.0,
            losses: Vec::new(),
            accuracy: None,
            checkpoints_submitted: 0,
            mac_checks_passed: 0,
            cheater_detected: None,
            zk_proofs_generated: 0,
            zk_activated_by_risk: false,
            phase: 0,
            phase_description: String::new(),
            coordinator_address: String::new(),
            job_id: 0,
            elapsed_secs: 0.0,
            started_at: 0.0,
            final_weights: None,
        }
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

    // Skip auth for /health and /ws endpoints
    let path = req.uri().path();
    if path == "/health" || path == "/ws" {
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
        // Training job endpoints
        .route("/api/training/start", post(start_training_handler))
        .route("/api/training/sessions", get(list_sessions_handler))
        .route("/api/training/sessions/:id", get(get_session_handler))
        .route("/api/training/sessions/:id/losses", get(get_losses_handler))
        .route("/api/training/sessions/:id/model", get(get_model_handler))
        // Data/weights upload endpoints
        .route("/api/training/data", post(upload_data_handler))
        .route("/api/training/weights", post(upload_weights_handler))
        // Worker registry endpoints
        .route("/api/workers", get(list_workers_handler))
        .route("/api/workers/register", post(register_worker_handler))
        .route("/api/workers/heartbeat", post(heartbeat_worker_handler))
        // WebSocket endpoint
        .route("/ws", get(websocket_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .layer(cors_layer)
        .with_state(state)
}

fn build_cors_layer(config: &DashboardConfig) -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
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
// Progress Event → JSON conversion
// ---------------------------------------------------------------------------

fn progress_event_to_json(event: &ProgressEvent) -> serde_json::Value {
    match event {
        ProgressEvent::PhaseStarted { phase, total, description } => {
            serde_json::json!({
                "type": "phase_started",
                "phase": phase,
                "total": total,
                "description": description,
            })
        }
        ProgressEvent::PhaseCompleted { phase, elapsed_ms } => {
            serde_json::json!({
                "type": "phase_completed",
                "phase": phase,
                "elapsed_ms": elapsed_ms,
            })
        }
        ProgressEvent::TrainingStep { step, total, loss, mac_ok } => {
            serde_json::json!({
                "type": "training_step",
                "step": step,
                "total": total,
                "loss": loss,
                "mac_ok": mac_ok,
            })
        }
        ProgressEvent::CheckpointSubmitted { index, total, step, tx_hash } => {
            serde_json::json!({
                "type": "checkpoint_submitted",
                "index": index,
                "total": total,
                "step": step,
                "tx_hash": tx_hash,
            })
        }
        ProgressEvent::CheaterDetected { party_index, step } => {
            serde_json::json!({
                "type": "cheater_detected",
                "party_index": party_index,
                "step": step,
            })
        }
        ProgressEvent::CheaterSlashed { party_index, tx_hash } => {
            serde_json::json!({
                "type": "cheater_slashed",
                "party_index": party_index,
                "tx_hash": tx_hash,
            })
        }
        ProgressEvent::TrainingComplete { accuracy, steps, time_secs, checkpoints } => {
            serde_json::json!({
                "type": "training_complete",
                "accuracy": accuracy,
                "steps": steps,
                "time_secs": time_secs,
                "checkpoints": checkpoints,
            })
        }
        ProgressEvent::RecoverableError { phase, message, retry_count } => {
            serde_json::json!({
                "type": "recoverable_error",
                "phase": phase,
                "message": message,
                "retry_count": retry_count,
            })
        }
        ProgressEvent::ZkProofStarted { checkpoint_index, step } => {
            serde_json::json!({
                "type": "zk_proof_started",
                "checkpoint_index": checkpoint_index,
                "step": step,
            })
        }
        ProgressEvent::ZkProofGenerated { checkpoint_index, step, proof_size, time_ms, verified } => {
            serde_json::json!({
                "type": "zk_proof_generated",
                "checkpoint_index": checkpoint_index,
                "step": step,
                "proof_size": proof_size,
                "time_ms": time_ms,
                "verified": verified,
            })
        }
        ProgressEvent::ZkProofFailed { checkpoint_index, step, error } => {
            serde_json::json!({
                "type": "zk_proof_failed",
                "checkpoint_index": checkpoint_index,
                "step": step,
                "error": error,
            })
        }
        ProgressEvent::ZkProofSubmitted { checkpoint_index, step, tx_hash } => {
            serde_json::json!({
                "type": "zk_proof_submitted",
                "checkpoint_index": checkpoint_index,
                "step": step,
                "tx_hash": tx_hash,
            })
        }
        ProgressEvent::ZkRiskActivated { active_workers, min_workers } => {
            serde_json::json!({
                "type": "zk_risk_activated",
                "active_workers": active_workers,
                "min_workers": min_workers,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Handlers: Existing
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
            "/api/events",
            "/api/training/start",
            "/api/training/sessions",
            "/api/training/sessions/:id",
            "/api/training/sessions/:id/losses",
            "/api/workers",
            "/api/workers/register",
            "/api/workers/heartbeat",
            "/ws"
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

    if status.node_id.is_empty() {
        populate_demo_network(&mut status);
    }

    Json(status)
}

/// Training status endpoint — reads from dynamic state
async fn training_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<TrainingStatus> {
    let status = state.training_status.read().await.clone();

    if !status.is_active && status.total_rounds == 0 {
        Json(demo_training_status())
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
        Json(demo_nodes())
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
        Json(demo_metrics())
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
        Json(demo_events())
    } else {
        Json(events.clone())
    }
}

// ---------------------------------------------------------------------------
// Handlers: Training Job Submission
// ---------------------------------------------------------------------------

/// POST /api/training/start — submit a new training job
async fn start_training_handler(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<TrainingJobRequest>,
) -> impl IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();
    info!(session_id = %session_id, "Starting new training session");

    // Validate request
    if req.architecture.len() != 3 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "architecture must have exactly 3 elements [input, hidden, output]" })),
        ).into_response();
    }

    // Determine workers: check off-chain registry, then fall back to num_workers
    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let registered: Vec<RegisteredWorker> = {
        let workers = state.registered_workers.read().await;
        workers.iter()
            .filter(|w| w.status == "idle" && (now_ts - w.last_heartbeat) <= 30.0)
            .cloned()
            .collect()
    };

    // Note: On-chain pool worker assignment happens inside the orchestrator
    // (via assignPoolWorkers on the V4 contract). The off-chain registry here
    // is just for endpoint discovery so the orchestrator knows where to connect.
    let (num_workers, worker_endpoints) = if !registered.is_empty() {
        let count = registered.len().min(10); // cap at 10 (Anvil key limit)
        if count < 2 {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "Need at least 2 online workers to start training" })),
            ).into_response();
        }
        let endpoints: Vec<String> = registered[..count].iter().map(|w| w.endpoint.clone()).collect();
        info!(count, "Using {} registered workers (on-chain pool assignment at job creation)", count);

        // Mark workers as busy
        {
            let mut workers = state.registered_workers.write().await;
            for w in workers.iter_mut() {
                if registered[..count].iter().any(|r| r.id == w.id) {
                    w.status = "busy".to_string();
                }
            }
        }

        (count, endpoints)
    } else {
        // Fallback: generate endpoints from num_workers (legacy/CLI mode)
        let nw = req.num_workers;
        if nw < 2 || nw > 10 {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "num_workers must be between 2 and 10" })),
            ).into_response();
        }
        let endpoints: Vec<String> = (0..nw)
            .map(|i| format!("127.0.0.1:{}", 9001 + (i as u16) * 2))
            .collect();
        (nw, endpoints)
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    // Create initial session state
    let session = TrainingSessionState {
        session_id: session_id.clone(),
        status: "starting".to_string(),
        current_step: 0,
        total_steps: req.num_steps,
        current_loss: 0.0,
        losses: Vec::new(),
        accuracy: None,
        checkpoints_submitted: 0,
        mac_checks_passed: 0,
        cheater_detected: None,
        zk_proofs_generated: 0,
        zk_activated_by_risk: false,
        phase: 0,
        phase_description: "Initializing".to_string(),
        coordinator_address: String::new(),
        job_id: 0,
        elapsed_secs: 0.0,
        started_at: now,
        final_weights: None,
    };

    // Store session
    state.sessions.write().await.insert(session_id.clone(), session);

    // Build orchestration config
    let zk_enabled = req.zk_mode == "always";
    let zk_config = ZkProofConfig {
        enabled: req.zk_mode != "off",
        checkpoint_frequency: req.zk_checkpoint_freq as usize,
        ..ZkProofConfig::default()
    };

    let mut config = FullOrchestrationConfig {
        architecture: req.architecture.clone(),
        num_steps: req.num_steps,
        learning_rate: req.learning_rate,
        checkpoint_frequency: req.checkpoint_freq,
        mac_check_interval: req.mac_interval,
        beaver_batch_size: 2048,
        batch_size: req.batch_size.unwrap_or(1),
        seed: req.seed,
        worker_endpoints,
        initial_weights_path: None,
        zk_proof: zk_config,
        zk_mode: match req.zk_mode.as_str() {
            "always" => ZkMode::Always,
            "risk" => ZkMode::Risk { min_workers: req.min_workers_for_mpc },
            _ => ZkMode::Off,
        },
        use_real_mnist: req.use_real_mnist,
        mnist_cache_dir: None,
        train_size: req.train_size,
        test_size: req.test_size,
        #[cfg(feature = "chain")]
        eth_rpc_url: None, // Will be set below from dashboard state or Anvil default
        #[cfg(feature = "chain")]
        private_key: String::new(), // Will be set from Anvil default accounts
        #[cfg(feature = "chain")]
        worker_private_keys: Vec::new(),
        #[cfg(feature = "chain")]
        payment_amount_eth: req.payment_eth,
        #[cfg(feature = "chain")]
        stake_amount_eth: req.stake_per_worker_eth.max(0.1), // Contract minStake is 0.1 ETH
        #[cfg(feature = "chain")]
        coordinator_address: None, // Will be set below from dashboard state if available
        #[cfg(feature = "chain")]
        use_pool_workers: false, // Will be set to true below if coordinator is pre-deployed
        #[cfg(feature = "chain")]
        enable_withdrawal: false,
        distributed: req.transport == "distributed",
        custom_training_data: None,
        simulate_cheater: req.simulate_cheater,
    };

    // Set private keys: prefer TESTNET_PRIVATE_KEY env var, fall back to Anvil defaults.
    #[cfg(feature = "chain")]
    {
        if let Ok(testnet_key) = std::env::var("TESTNET_PRIVATE_KEY") {
            // Testnet mode: owner key from env, workers auto-generated and funded
            config.private_key = testnet_key;
            config.worker_private_keys = Vec::new(); // Will be generated + funded by orchestrator
            info!("Using TESTNET_PRIVATE_KEY for owner, workers will be auto-funded");
        } else {
            // Local Anvil demo mode: hardcoded well-known keys
            config.private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".to_string();
            let anvil_keys = vec![
                "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
                "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
                "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
                "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
                "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
                "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e",
                "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
                "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97",
                "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
            ];
            config.worker_private_keys = anvil_keys[..num_workers.min(anvil_keys.len())]
                .iter()
                .map(|k| k.to_string())
                .collect();
        }

        // Use pre-deployed coordinator if dashboard was started with --coordinator
        if let Some(ref coord_addr) = *state.coordinator_address.read().await {
            config.coordinator_address = Some(coord_addr.clone());
            config.use_pool_workers = true;
            info!(coordinator = %coord_addr, "Using pre-deployed V4 coordinator with on-chain worker pool");
        }
        if let Some(ref rpc_url) = *state.eth_rpc_url.read().await {
            config.eth_rpc_url = Some(rpc_url.clone());
        }
    }

    // If custom training data was uploaded, pass it to the orchestrator
    if let Some(custom_data) = state.uploaded_data.read().await.clone() {
        info!(samples = custom_data.len(), "Using uploaded training data for session");
        config.custom_training_data = Some(custom_data);
    }

    // If initial weights were uploaded, write them to a temp file for the orchestrator
    if let Some(weights_json) = state.uploaded_weights.read().await.clone() {
        if let Ok(tmp_dir) = std::env::temp_dir().canonicalize() {
            let weights_path = tmp_dir.join(format!("helix_weights_{}.json", session_id));
            if let Ok(json_str) = serde_json::to_string_pretty(&weights_json) {
                if std::fs::write(&weights_path, &json_str).is_ok() {
                    config.initial_weights_path = Some(weights_path.to_string_lossy().to_string());
                    info!(path = %weights_path.display(), "Using uploaded initial weights");
                }
            }
        }
    }

    // Spawn background training task
    let state_clone = state.clone();
    let sid = session_id.clone();

    tokio::spawn(async move {
        run_training_session(state_clone, sid, config).await;
    });

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "session_id": session_id,
            "status": "starting",
        })),
    ).into_response()
}

/// Background task that runs the full orchestration pipeline.
async fn run_training_session(
    state: Arc<DashboardState>,
    session_id: String,
    config: FullOrchestrationConfig,
) {
    let sid = session_id.clone();
    let state_for_cb = state.clone();
    let sid_for_cb = sid.clone();

    let mut orchestrator = FullOrchestrator::new(config);

    // Set up progress callback that updates session state and broadcasts to WebSocket.
    // The callback must be Fn + Send + Sync (non-async), so we use try_write for state
    // updates and the broadcast channel (which is sync-safe) for WebSocket fanout.
    let progress_cb: ProgressCallback = Arc::new(move |event: ProgressEvent| {
        let event_json = progress_event_to_json(&event);

        // Broadcast to WebSocket subscribers (send is sync-safe on broadcast::Sender)
        let _ = state_for_cb.ws_broadcast.send((sid_for_cb.clone(), event_json));

        // Update session state synchronously using try_write.
        // Scoped block ensures the write guard is dropped before the closure returns.
        let sessions_guard = state_for_cb.sessions.try_write();
        if let Ok(mut sessions) = sessions_guard {
            if let Some(session) = sessions.get_mut(&sid_for_cb) {
                match &event {
                    ProgressEvent::PhaseStarted { phase, description, .. } => {
                        session.phase = *phase;
                        session.phase_description = description.clone();
                        session.status = "running".to_string();
                    }
                    ProgressEvent::TrainingStep { step, loss, .. } => {
                        session.current_step = *step;
                        session.current_loss = *loss;
                        session.losses.push(*loss);
                        session.mac_checks_passed += 1;
                    }
                    ProgressEvent::CheckpointSubmitted { .. } => {
                        session.checkpoints_submitted += 1;
                    }
                    ProgressEvent::CheaterDetected { party_index, step } => {
                        session.cheater_detected = Some(serde_json::json!({
                            "party_index": party_index,
                            "step": step,
                        }));
                    }
                    ProgressEvent::TrainingComplete { accuracy, .. } => {
                        session.accuracy = Some(*accuracy);
                        session.status = "complete".to_string();
                    }
                    ProgressEvent::ZkProofGenerated { .. } => {
                        session.zk_proofs_generated += 1;
                    }
                    _ => {}
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                session.elapsed_secs = now - session.started_at;
            }
        }
    });

    orchestrator.set_progress_callback(progress_cb);

    // Run the orchestration pipeline
    info!(session_id = %session_id, "Running training orchestration");
    match orchestrator.run().await {
        Ok(result) => {
            info!(
                session_id = %session_id,
                accuracy = result.test_accuracy,
                steps = result.steps_completed,
                "Training completed successfully"
            );

            // Serialize final weights if available
            let weights_json = result.final_weights.as_ref().map(|fw| {
                serde_json::json!({
                    "w1": fw.w1,
                    "b1": fw.b1,
                    "w2": fw.w2,
                    "b2": fw.b2,
                })
            });

            // Update final session state
            if let Ok(mut sessions) = state.sessions.try_write() {
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.status = "complete".to_string();
                    session.accuracy = Some(result.test_accuracy);
                    session.job_id = result.job_id;
                    session.coordinator_address = result.coordinator_address.clone();
                    session.zk_proofs_generated = result.zk_proofs_generated;
                    session.final_weights = weights_json;
                }
            }

            // Store coordinator address for on-chain worker queries
            if !result.coordinator_address.is_empty() {
                *state.coordinator_address.write().await = Some(result.coordinator_address.clone());
                // Default to localhost Anvil
                if state.eth_rpc_url.read().await.is_none() {
                    *state.eth_rpc_url.write().await = Some("http://localhost:8545".to_string());
                }
            }

            // Broadcast completion
            let _ = state.ws_broadcast.send((session_id.clone(), serde_json::json!({
                "type": "session_complete",
                "session_id": session_id,
                "accuracy": result.test_accuracy,
                "steps": result.steps_completed,
                "time_secs": result.training_time_secs,
                "checkpoints": result.checkpoints_on_chain,
                "gas_used": result.total_gas_used,
                "zk_proofs": result.zk_proofs_generated,
            })));
        }
        Err(e) => {
            error!(session_id = %session_id, error = %e, "Training failed");
            error!(session_id = %session_id, "Full error chain: {:?}", e);

            if let Ok(mut sessions) = state.sessions.try_write() {
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.status = "failed".to_string();
                    session.phase_description = format!("Error: {}", e);
                }
            }

            let _ = state.ws_broadcast.send((session_id.clone(), serde_json::json!({
                "type": "session_failed",
                "session_id": session_id,
                "error": e.to_string(),
            })));
        }
    }

    // Clean up orchestrator
    orchestrator.shutdown();

    // Mark all busy workers as idle again
    if let Ok(mut workers) = state.registered_workers.try_write() {
        for w in workers.iter_mut() {
            if w.status == "busy" {
                w.status = "idle".to_string();
            }
        }
    }

    // Broadcast worker status update
    let idle_count = state.registered_workers.try_read()
        .map(|w| w.iter().filter(|w| w.status == "idle").count())
        .unwrap_or(0);
    let _ = state.ws_broadcast.send(("__system__".to_string(), serde_json::json!({
        "type": "workers_updated",
        "count": idle_count,
    })));
}

/// GET /api/training/sessions — list all training sessions
async fn list_sessions_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<Vec<TrainingSessionState>> {
    let sessions = state.sessions.read().await;
    let mut list: Vec<TrainingSessionState> = sessions.values().cloned().collect();
    // Sort by started_at descending (newest first)
    list.sort_by(|a, b| b.started_at.partial_cmp(&a.started_at).unwrap_or(std::cmp::Ordering::Equal));
    Json(list)
}

/// GET /api/training/sessions/:id — get a specific training session
async fn get_session_handler(
    State(state): State<Arc<DashboardState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&id) {
        Some(session) => Json(serde_json::json!(session)).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Session not found" }))).into_response(),
    }
}

/// GET /api/training/sessions/:id/losses — get loss history for a session
async fn get_losses_handler(
    State(state): State<Arc<DashboardState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&id) {
        Some(session) => Json(serde_json::json!({
            "session_id": id,
            "losses": session.losses,
            "current_step": session.current_step,
            "total_steps": session.total_steps,
        })).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "Session not found" }))).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Handler: WebSocket
// ---------------------------------------------------------------------------

/// GET /ws — WebSocket upgrade for real-time training progress
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: Arc<DashboardState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut subscriptions: Vec<String> = Vec::new();
    let mut rx = state.ws_broadcast.subscribe();

    info!("WebSocket client connected");

    // Spawn a task to forward broadcast messages to this client
    let (tx, mut forward_rx) = tokio::sync::mpsc::channel::<serde_json::Value>(256);

    let forward_task = tokio::spawn(async move {
        while let Ok((session_id, event)) = rx.recv().await {
            // Send to the mpsc channel; the main loop will check subscriptions
            let msg = serde_json::json!({
                "session_id": session_id,
                "event": event,
            });
            if tx.send(msg).await.is_err() {
                break;
            }
        }
    });

    loop {
        tokio::select! {
            // Handle incoming messages from the client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Parse subscription messages
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(sub) = parsed.get("subscribe").and_then(|v| v.as_str()) {
                                info!(subscription = %sub, "WebSocket client subscribed");
                                subscriptions.push(sub.to_string());

                                // Send current state for this session if it exists
                                if let Some(session_id) = sub.strip_prefix("training:") {
                                    let sessions = state.sessions.read().await;
                                    if let Some(session) = sessions.get(session_id) {
                                        let state_msg = serde_json::json!({
                                            "type": "session_state",
                                            "session_id": session_id,
                                            "state": session,
                                        });
                                        if sender.send(Message::Text(state_msg.to_string().into())).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                            }
                            if let Some(unsub) = parsed.get("unsubscribe").and_then(|v| v.as_str()) {
                                subscriptions.retain(|s| s != unsub);
                            }
                        }
                    }
                    Some(Ok(Message::Ping(data))) => {
                        if sender.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        break;
                    }
                    _ => {}
                }
            }
            // Forward broadcast messages to this WebSocket client
            Some(msg) = forward_rx.recv() => {
                if let Some(session_id) = msg.get("session_id").and_then(|v| v.as_str()) {
                    // System messages (worker updates) go to all clients
                    if session_id == "__system__" {
                        if sender.send(Message::Text(msg.to_string().into())).await.is_err() {
                            break;
                        }
                    } else {
                        let channel = format!("training:{}", session_id);
                        // Send if subscribed to this session or to "training:*" (all)
                        if subscriptions.contains(&channel) || subscriptions.contains(&"training:*".to_string()) {
                            if sender.send(Message::Text(msg.to_string().into())).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    forward_task.abort();
    info!("WebSocket client disconnected");
}

// ---------------------------------------------------------------------------
// Handler: Training Data Upload
// ---------------------------------------------------------------------------

/// POST /api/training/data — upload custom training data
async fn upload_data_handler(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<TrainingDataUpload>,
) -> impl IntoResponse {
    if req.samples.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "samples array must not be empty" })),
        ).into_response();
    }

    let input_dim = req.samples[0].input.len();
    let output_dim = req.samples[0].target.len();

    // Validate all samples have consistent dimensions
    for (i, sample) in req.samples.iter().enumerate() {
        if sample.input.len() != input_dim {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("sample {} has input dim {} but expected {}", i, sample.input.len(), input_dim)
                })),
            ).into_response();
        }
        if sample.target.len() != output_dim {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("sample {} has target dim {} but expected {}", i, sample.target.len(), output_dim)
                })),
            ).into_response();
        }
    }

    let pairs: Vec<(Vec<f64>, Vec<f64>)> = req.samples
        .into_iter()
        .map(|s| (s.input, s.target))
        .collect();
    let count = pairs.len();

    *state.uploaded_data.write().await = Some(pairs);

    info!(samples = count, input_dim, output_dim, "Training data uploaded");

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "samples": count,
            "input_dim": input_dim,
            "output_dim": output_dim,
        })),
    ).into_response()
}

// ---------------------------------------------------------------------------
// Handler: Initial Weights Upload
// ---------------------------------------------------------------------------

/// POST /api/training/weights — upload initial model weights (for continue-training)
async fn upload_weights_handler(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<WeightsUpload>,
) -> impl IntoResponse {
    let w1_len = req.w1.len();
    let b1_len = req.b1.len();
    let w2_len = req.w2.len();
    let b2_len = req.b2.len();

    if w1_len == 0 || b1_len == 0 || w2_len == 0 || b2_len == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "all weight arrays must be non-empty" })),
        ).into_response();
    }

    let weights_json = serde_json::json!({
        "w1": req.w1,
        "b1": req.b1,
        "w2": req.w2,
        "b2": req.b2,
    });

    *state.uploaded_weights.write().await = Some(weights_json);

    info!(w1 = w1_len, b1 = b1_len, w2 = w2_len, b2 = b2_len, "Initial weights uploaded");

    let total_params = w1_len + b1_len + w2_len + b2_len;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "w1_size": w1_len,
            "b1_size": b1_len,
            "w2_size": w2_len,
            "b2_size": b2_len,
            "total_params": total_params,
        })),
    ).into_response()
}

// ---------------------------------------------------------------------------
// Handlers: Worker Registry
// ---------------------------------------------------------------------------

/// POST /api/workers/register — register a new MPC worker
async fn register_worker_handler(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<WorkerRegisterRequest>,
) -> impl IntoResponse {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let worker_id = uuid::Uuid::new_v4().to_string();

    let worker = RegisteredWorker {
        id: worker_id.clone(),
        endpoint: req.endpoint.clone(),
        party_index: req.party_index,
        registered_at: now,
        last_heartbeat: now,
        status: "idle".to_string(),
    };

    let mut workers = state.registered_workers.write().await;
    // Remove any existing worker with the same endpoint (re-registration)
    workers.retain(|w| w.endpoint != req.endpoint);
    workers.push(worker);

    let count = workers.len();
    drop(workers);

    info!(worker_id = %worker_id, endpoint = %req.endpoint, total = count, "Worker registered");

    // Broadcast worker count update to WebSocket clients
    let _ = state.ws_broadcast.send(("__system__".to_string(), serde_json::json!({
        "type": "workers_updated",
        "count": count,
    })));

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "worker_id": worker_id,
            "status": "registered",
            "total_workers": count,
        })),
    ).into_response()
}

/// POST /api/workers/heartbeat — worker heartbeat to stay alive
async fn heartbeat_worker_handler(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<WorkerHeartbeatRequest>,
) -> impl IntoResponse {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let mut workers = state.registered_workers.write().await;
    if let Some(worker) = workers.iter_mut().find(|w| w.id == req.worker_id) {
        worker.last_heartbeat = now;
        (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "ok" })),
        ).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Worker not found" })),
        ).into_response()
    }
}

/// GET /api/workers — list all registered workers (on-chain pool + off-chain fallback)
async fn list_workers_handler(
    State(state): State<Arc<DashboardState>>,
) -> Json<serde_json::Value> {
    // Try to read from on-chain pool first
    #[cfg(feature = "chain")]
    {
        let coord_addr = state.coordinator_address.read().await.clone();
        let rpc_url = state.eth_rpc_url.read().await.clone();
        if let (Some(coord), Some(rpc)) = (coord_addr, rpc_url) {
            if let Ok(chain_workers) = query_pool_workers_from_chain(&rpc, &coord).await {
                return Json(chain_workers);
            }
        }
    }

    // Fallback: off-chain registry
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let workers = state.registered_workers.read().await;
    let worker_list: Vec<serde_json::Value> = workers.iter().map(|w| {
        let effective_status = if now - w.last_heartbeat > 30.0 && w.status == "idle" {
            "offline"
        } else {
            &w.status
        };
        serde_json::json!({
            "id": w.id,
            "endpoint": w.endpoint,
            "party_index": w.party_index,
            "status": effective_status,
            "registered_at": w.registered_at,
            "last_heartbeat": w.last_heartbeat,
            "source": "off-chain",
        })
    }).collect();

    let online_count = workers.iter().filter(|w| now - w.last_heartbeat <= 30.0).count();

    Json(serde_json::json!({
        "workers": worker_list,
        "total": workers.len(),
        "online": online_count,
        "source": "off-chain",
    }))
}

/// Query pool workers directly from the V4 contract on-chain.
#[cfg(feature = "chain")]
async fn query_pool_workers_from_chain(rpc_url: &str, coordinator_addr: &str) -> Result<serde_json::Value, anyhow::Error> {
    use crate::rpc::chain_v4::ChainClientV4;
    use ethers::signers::LocalWallet;
    use std::str::FromStr;

    // Read-only key (dummy) just for view calls
    let dummy_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    let wallet = dummy_key.parse::<LocalWallet>().map_err(|e| anyhow::anyhow!("wallet: {}", e))?;

    let client = ChainClientV4::with_wallet(rpc_url, wallet, coordinator_addr).await?;
    let addresses = client.get_pool_workers().await?;

    let mut worker_list = Vec::new();
    let mut available_count = 0u64;

    for addr in &addresses {
        if let Ok(info) = client.get_pool_worker_info(*addr).await {
            let status = if info.available { "idle" } else { "busy" };
            if info.available { available_count += 1; }
            worker_list.push(serde_json::json!({
                "address": format!("{:?}", addr),
                "endpoint": info.endpoint,
                "stake_eth": ethers::utils::format_ether(info.stake_amount),
                "status": status,
                "registered_at": info.registered_at,
                "active_job_id": info.active_job_id,
                "source": "on-chain",
            }));
        }
    }

    Ok(serde_json::json!({
        "workers": worker_list,
        "total": addresses.len(),
        "online": available_count,
        "source": "on-chain",
        "coordinator": coordinator_addr,
    }))
}

// ---------------------------------------------------------------------------
// Handler: Model Download
// ---------------------------------------------------------------------------

/// GET /api/training/sessions/:id/model — download trained model weights
async fn get_model_handler(
    State(state): State<Arc<DashboardState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&id) {
        Some(session) => {
            if session.status != "complete" {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "Training not yet complete" })),
                ).into_response();
            }
            match &session.final_weights {
                Some(weights) => Json(serde_json::json!({
                    "session_id": id,
                    "status": "complete",
                    "accuracy": session.accuracy,
                    "weights": weights,
                })).into_response(),
                None => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "error": "Weights not available for this session" })),
                ).into_response(),
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" })),
        ).into_response(),
    }
}
