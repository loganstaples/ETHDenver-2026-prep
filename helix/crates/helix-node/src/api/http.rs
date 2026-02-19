//! HTTP API for aggregator node monitoring and control.
//!
//! Endpoints:
//! - `GET /health` — public, no auth required
//! - `GET /metrics` — public, no auth required
//! - `POST /round/start` — requires Bearer token
//! - `GET /round/status` — requires Bearer token
//! - `GET /workers` — requires Bearer token
//! - `GET /peers` — requires Bearer token

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Path, State};
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::api::rpc::RoundWeightEntry;
use crate::metrics::NodeMetrics;
use crate::round_commit::RoundCommitManager;
use crate::training::orchestrator::{RoundPhase, TrainingOrchestrator};

// ---------------------------------------------------------------------------
// Token bucket rate limiter (per-IP)
// ---------------------------------------------------------------------------

/// Simple token-bucket rate limiter keyed by IP address.
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    /// Attempts to consume one token. Returns true if allowed.
    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Thread-safe per-IP rate limiter.
pub struct ApiRateLimiter {
    buckets: parking_lot::Mutex<HashMap<IpAddr, TokenBucket>>,
    capacity: f64,
    refill_rate: f64,
}

impl ApiRateLimiter {
    pub fn new(requests_per_sec: u32) -> Self {
        let rate = requests_per_sec as f64;
        Self {
            buckets: parking_lot::Mutex::new(HashMap::new()),
            // Allow small burst (2x rate), refill at configured rate
            capacity: rate * 2.0,
            refill_rate: rate,
        }
    }

    /// Returns true if the request from `ip` is allowed.
    pub fn check(&self, ip: IpAddr) -> bool {
        let mut buckets = self.buckets.lock();
        let bucket = buckets
            .entry(ip)
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_rate));
        bucket.try_consume()
    }

    /// Removes expired buckets that haven't been used recently.
    pub fn cleanup(&self) {
        let mut buckets = self.buckets.lock();
        let cutoff = Instant::now() - std::time::Duration::from_secs(300);
        buckets.retain(|_, b| b.last_refill > cutoff);
    }
}

// ---------------------------------------------------------------------------
// Shared API state
// ---------------------------------------------------------------------------

/// Shared state for the HTTP API.
pub struct ApiState {
    /// Reference to the orchestrator (for worker/round info).
    pub orchestrator_workers: Arc<RwLock<OrchestratorSnapshot>>,
    /// Trigger channel for starting rounds.
    pub round_trigger_tx: broadcast::Sender<()>,
    /// Snapshot of connected peers (updated periodically).
    pub peers: Arc<RwLock<PeerSnapshot>>,
    /// Snapshot of aggregate metrics (updated periodically).
    pub metrics: Arc<RwLock<MetricsSnapshot>>,
    /// Node-wide atomic metrics (for Prometheus endpoint and RPC).
    pub node_metrics: Arc<NodeMetrics>,
    /// Bearer token required for authenticated endpoints.
    /// Must match the `Authorization: Bearer <token>` header.
    pub api_key: String,
    /// Per-IP rate limiter.
    pub rate_limiter: Arc<ApiRateLimiter>,
    /// Completed round weights for result distribution (shared with RPC layer).
    pub round_weights: Arc<RwLock<Vec<RoundWeightEntry>>>,
    /// Node role for health endpoint.
    pub node_role: String,
    /// Start time for uptime calculation.
    pub start_time: Instant,
    /// Last checkpoint timestamp (epoch seconds).
    pub last_checkpoint_ts: Arc<RwLock<Option<u64>>>,
    /// MPC health status (updated by main loop).
    pub mpc_health: Arc<RwLock<MpcHealthStatus>>,
    /// Fault tolerance status (updated by main loop).
    pub fault_tolerance_status: Arc<RwLock<FaultToleranceStatus>>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Snapshot of orchestrator state (updated periodically).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrchestratorSnapshot {
    pub worker_count: usize,
    pub available_workers: usize,
    pub computing_workers: usize,
    pub current_round: Option<RoundInfo>,
    pub completed_rounds: u64,
    pub workers: Vec<WorkerInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundInfo {
    pub round_id: u64,
    pub phase: String,
    pub gradients_received: usize,
    pub workers_assigned: usize,
    /// Current model commitment hash (hex-encoded, e.g. "0xabcd...").
    /// `None` if no commitment has been computed yet.
    pub commitment_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
    pub rounds_participated: u64,
    pub cpu_load: u8,
    pub memory_mb: u64,
    pub reputation_score: f64,
    pub success_rate: f64,
    pub earnings_wei: u64,
    pub last_heartbeat: u64,
    pub capabilities: WorkerCapabilities,
}

impl WorkerInfo {
    /// Create a WorkerInfo with only id, status, and rounds_completed set;
    /// all other fields use sensible defaults.
    pub fn new(id: String, status: String, rounds_completed: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            status,
            rounds_participated: rounds_completed,
            cpu_load: 0,
            memory_mb: 0,
            reputation_score: 0.5,
            success_rate: if rounds_completed > 0 { 1.0 } else { 0.0 },
            earnings_wei: 0,
            last_heartbeat: now,
            rounds_completed,
            capabilities: WorkerCapabilities::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkerCapabilities {
    pub can_train: bool,
    pub can_prove: bool,
    pub can_aggregate: bool,
    pub gpu_model: Option<String>,
    pub max_batch_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerEntry {
    pub id: String,
    pub address: String,
    pub last_seen: u64,
    pub reputation: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerSnapshot {
    pub peers: Vec<PeerEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub total_rounds: u64,
    pub total_proofs: u64,
    pub proofs_valid: u64,
    pub proofs_invalid: u64,
    pub avg_round_time_ms: f64,
    pub uptime_secs: u64,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            total_rounds: 0,
            total_proofs: 0,
            proofs_valid: 0,
            proofs_invalid: 0,
            avg_round_time_ms: 0.0,
            uptime_secs: 0,
        }
    }
}

/// Rich health response with training progress, MPC status, memory, etc.
#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub role: String,
    pub workers: usize,
    pub current_round: Option<u64>,
    /// Training progress: total completed rounds.
    pub completed_rounds: u64,
    /// Number of connected peers.
    pub connected_peers: usize,
    /// Uptime in seconds since the API state was created.
    pub uptime_secs: u64,
    /// MPC session status.
    pub mpc: MpcHealthStatus,
    /// Last checkpoint timestamp (epoch seconds), if any.
    pub last_checkpoint_ts: Option<u64>,
    /// Resident memory usage in bytes (macOS/Linux).
    pub memory_bytes: u64,
    /// Node fault tolerance status.
    pub fault_tolerance: FaultToleranceStatus,
}

/// MPC health sub-status.
#[derive(Serialize, Deserialize, Default)]
pub struct MpcHealthStatus {
    pub enabled: bool,
    pub active_session: bool,
    pub session_id: Option<String>,
    pub num_parties: usize,
    pub party_index: Option<usize>,
}

/// Fault tolerance sub-status.
#[derive(Serialize, Deserialize, Default)]
pub struct FaultToleranceStatus {
    pub healthy_workers: usize,
    pub degraded_workers: usize,
    pub failed_workers: usize,
    pub system_healthy: bool,
}

#[derive(Serialize, Deserialize)]
struct RoundStartResponse {
    triggered: bool,
    message: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Middleware: per-IP rate limiting
// ---------------------------------------------------------------------------

async fn rate_limit_middleware(
    State(state): State<Arc<ApiState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !state.rate_limiter.check(addr.ip()) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: "Rate limit exceeded".to_string(),
            }),
        )
            .into_response();
    }
    next.run(request).await
}

// ---------------------------------------------------------------------------
// Middleware: Bearer token authentication (mutating endpoints only)
// ---------------------------------------------------------------------------

async fn auth_middleware(
    State(state): State<Arc<ApiState>>,
    headers: HeaderMap,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Extract Bearer token
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match token {
        Some(t) if constant_time_eq(t.as_bytes(), state.api_key.as_bytes()) => {
            next.run(request).await
        }
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Missing or invalid Bearer token".to_string(),
            }),
        )
            .into_response(),
    }
}

/// Constant-time byte comparison to prevent timing side-channel attacks
/// on API key validation.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    a.ct_eq(b).into()
}

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

/// Starts the HTTP API server on the given address.
///
/// Public endpoints (`/health`, `/metrics`) require no authentication.
/// All other endpoints require a `Bearer <api_key>` token in the
/// `Authorization` header.
///
/// All endpoints are subject to per-IP token-bucket rate limiting.
#[tracing::instrument(skip_all)]
pub async fn start_api_server(
    addr: SocketAddr,
    state: Arc<ApiState>,
) -> anyhow::Result<()> {
    // Public routes (no auth)
    let public_routes = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler));

    // Authenticated routes (require Bearer token)
    let auth_routes = Router::new()
        .route("/round/start", post(round_start_handler))
        .route("/round/status", get(round_status_handler))
        .route("/workers", get(workers_handler))
        .route("/peers", get(peers_handler))
        .route("/models/{model_id}/rounds/{round_id}/weights", get(download_weights_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Combine and add rate limiting to everything
    let app = Router::new()
        .merge(public_routes)
        .merge(auth_routes)
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("HTTP API listening on {}", addr);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health_handler(State(state): State<Arc<ApiState>>) -> Json<HealthResponse> {
    let snapshot = state.orchestrator_workers.read();
    let peers = state.peers.read();
    let ft = state.fault_tolerance_status.read();

    // Determine overall status from fault tolerance
    let status = if ft.system_healthy {
        "healthy".to_string()
    } else if ft.failed_workers > 0 {
        "degraded".to_string()
    } else {
        "healthy".to_string()
    };

    Json(HealthResponse {
        status,
        role: state.node_role.clone(),
        workers: snapshot.worker_count,
        current_round: snapshot.current_round.as_ref().map(|r| r.round_id),
        completed_rounds: snapshot.completed_rounds,
        connected_peers: peers.peers.len(),
        uptime_secs: state.start_time.elapsed().as_secs(),
        mpc: {
            let mpc = state.mpc_health.read();
            MpcHealthStatus {
                enabled: mpc.enabled,
                active_session: mpc.active_session,
                session_id: mpc.session_id.clone(),
                num_parties: mpc.num_parties,
                party_index: mpc.party_index,
            }
        },
        last_checkpoint_ts: *state.last_checkpoint_ts.read(),
        memory_bytes: get_resident_memory(),
        fault_tolerance: FaultToleranceStatus {
            healthy_workers: ft.healthy_workers,
            degraded_workers: ft.degraded_workers,
            failed_workers: ft.failed_workers,
            system_healthy: ft.system_healthy,
        },
    })
}

/// Returns the resident memory size of this process in bytes.
fn get_resident_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        // macOS: use mach task_info
        extern "C" {
            fn mach_task_self() -> u32;
        }
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }
        unsafe {
            let mut info: MachTaskBasicInfo = std::mem::zeroed();
            let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
            let ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO, &mut info, &mut count);
            if ret == 0 {
                return info.resident_size;
            }
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        // Linux: read /proc/self/statm
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    return pages * 4096; // page size
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

async fn round_start_handler(State(state): State<Arc<ApiState>>) -> Json<RoundStartResponse> {
    match state.round_trigger_tx.send(()) {
        Ok(_) => Json(RoundStartResponse {
            triggered: true,
            message: "Round start triggered".to_string(),
        }),
        Err(_) => Json(RoundStartResponse {
            triggered: false,
            message: "No listeners for round trigger".to_string(),
        }),
    }
}

async fn round_status_handler(State(state): State<Arc<ApiState>>) -> Json<OrchestratorSnapshot> {
    let snapshot = state.orchestrator_workers.read().clone();
    Json(snapshot)
}

async fn workers_handler(State(state): State<Arc<ApiState>>) -> Json<Vec<WorkerInfo>> {
    let snapshot = state.orchestrator_workers.read();
    Json(snapshot.workers.clone())
}

async fn peers_handler(State(state): State<Arc<ApiState>>) -> Json<PeerSnapshot> {
    let snapshot = state.peers.read().clone();
    Json(snapshot)
}

async fn metrics_handler(State(state): State<Arc<ApiState>>) -> Response {
    let body = state.node_metrics.to_prometheus();
    (
        StatusCode::OK,
        [(
            "content-type",
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// Downloads model weights for a completed round as raw bytes.
///
/// Returns the serialized weights with commitment and metadata headers.
async fn download_weights_handler(
    State(state): State<Arc<ApiState>>,
    Path((model_id, round_id)): Path<(u64, u64)>,
) -> Response {
    let weights = state.round_weights.read();
    let entry = weights.iter().find(|w| w.model_id == model_id && w.round_id == round_id);

    match entry {
        Some(e) => {
            let commitment_hex = format!("0x{}", hex::encode(e.commitment));
            (
                StatusCode::OK,
                [
                    ("content-type", "application/octet-stream"),
                    ("content-disposition", &format!("attachment; filename=\"model_{}_round_{}.weights\"", model_id, round_id)),
                ],
                [
                    ("x-weight-commitment", commitment_hex.as_str()),
                    ("x-round-id", &round_id.to_string()),
                    ("x-model-id", &model_id.to_string()),
                    ("x-loss", &e.loss.to_string()),
                    ("x-error-bound", &e.error_bound.to_string()),
                ],
                e.weight_bytes.clone(),
            ).into_response()
        }
        None => {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Weights not found for model {} round {}", model_id, round_id),
                }),
            ).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for oneshot

    fn test_state(api_key: &str) -> Arc<ApiState> {
        let (tx, _rx) = broadcast::channel(16);
        let node_metrics = NodeMetrics::new();
        // Pre-populate some metrics for testing
        node_metrics.record_proof_generated(100);
        node_metrics.record_proof_verified();
        node_metrics.record_proof_verified();
        node_metrics.record_training_step(0.42);
        Arc::new(ApiState {
            orchestrator_workers: Arc::new(RwLock::new(OrchestratorSnapshot {
                worker_count: 5,
                available_workers: 3,
                computing_workers: 2,
                current_round: Some(RoundInfo {
                    round_id: 42,
                    phase: "Collecting".to_string(),
                    gradients_received: 2,
                    workers_assigned: 5,
                    commitment_hash: Some("0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890".to_string()),
                }),
                completed_rounds: 10,
                workers: vec![
                    WorkerInfo::new("w1".to_string(), "Computing".to_string(), 5),
                ],
            })),
            round_trigger_tx: tx,
            peers: Arc::new(RwLock::new(PeerSnapshot {
                peers: vec![PeerEntry {
                    id: "peer1".to_string(),
                    address: "127.0.0.1:9000".to_string(),
                    last_seen: 12345,
                    reputation: 50,
                }],
            })),
            metrics: Arc::new(RwLock::new(MetricsSnapshot {
                total_rounds: 100,
                total_proofs: 200,
                proofs_valid: 190,
                proofs_invalid: 10,
                avg_round_time_ms: 5000.0,
                uptime_secs: 3600,
            })),
            node_metrics,
            api_key: api_key.to_string(),
            rate_limiter: Arc::new(ApiRateLimiter::new(100)),
            round_weights: Arc::new(RwLock::new(Vec::new())),
            node_role: "aggregator".to_string(),
            start_time: Instant::now(),
            last_checkpoint_ts: Arc::new(RwLock::new(Some(1700000000))),
            mpc_health: Arc::new(RwLock::new(MpcHealthStatus {
                enabled: true,
                active_session: true,
                session_id: Some("test-session".to_string()),
                num_parties: 3,
                party_index: Some(0),
            })),
            fault_tolerance_status: Arc::new(RwLock::new(FaultToleranceStatus {
                healthy_workers: 4,
                degraded_workers: 1,
                failed_workers: 0,
                system_healthy: true,
            })),
        })
    }

    /// Build the app router without ConnectInfo (uses a simpler setup for tests).
    fn test_app(state: Arc<ApiState>) -> Router {
        // For testing, we skip the rate-limit middleware since it requires ConnectInfo.
        // We test rate limiting separately.
        let public_routes = Router::new()
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler));

        let auth_routes = Router::new()
            .route("/round/start", post(round_start_handler))
            .route("/round/status", get(round_status_handler))
            .route("/workers", get(workers_handler))
            .route("/peers", get(peers_handler))
            .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

        Router::new()
            .merge(public_routes)
            .merge(auth_routes)
            .with_state(state)
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = test_state("test-key");
        let app = test_app(state);

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.status, "healthy");
        assert_eq!(health.role, "aggregator");
        assert_eq!(health.workers, 5);
        assert_eq!(health.current_round, Some(42));
        assert_eq!(health.completed_rounds, 10);
        assert_eq!(health.connected_peers, 1);
        assert!(health.uptime_secs < 5); // just created
        assert!(health.mpc.enabled);
        assert_eq!(health.mpc.num_parties, 3);
        assert!(health.last_checkpoint_ts.is_some());
        assert!(health.fault_tolerance.system_healthy);
        assert_eq!(health.fault_tolerance.healthy_workers, 4);
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let state = test_state("test-key");
        let app = test_app(state);

        let response = app
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // Verify Prometheus content-type
        let ct = response.headers().get("content-type").unwrap().to_str().unwrap();
        assert!(ct.contains("text/plain"), "expected text/plain content-type, got {}", ct);

        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        // Verify Prometheus format with pre-populated values from test_state
        assert!(body_str.contains("helix_proofs_generated_total 1"));
        assert!(body_str.contains("helix_proofs_verified_total 2"));
        assert!(body_str.contains("# TYPE helix_proofs_generated_total counter"));
        assert!(body_str.contains("# TYPE helix_uptime_seconds gauge"));
    }

    #[tokio::test]
    async fn test_auth_required_without_token() {
        let state = test_state("secret-token");
        let app = test_app(state);

        // Try accessing an auth-protected endpoint without a token
        let response = app
            .oneshot(Request::builder().uri("/workers").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_required_with_invalid_token() {
        let state = test_state("secret-token");
        let app = test_app(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/workers")
                    .header("authorization", "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_success_with_valid_token() {
        let state = test_state("secret-token");
        let app = test_app(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/workers")
                    .header("authorization", "Bearer secret-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let workers: Vec<WorkerInfo> = serde_json::from_slice(&body).unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, "w1");
    }

    #[tokio::test]
    async fn test_round_status_endpoint() {
        let state = test_state("my-key");
        let app = test_app(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/round/status")
                    .header("authorization", "Bearer my-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let snapshot: OrchestratorSnapshot = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.completed_rounds, 10);
        assert_eq!(snapshot.current_round.unwrap().round_id, 42);
    }

    #[tokio::test]
    async fn test_round_start_trigger() {
        let state = test_state("my-key");
        // Subscribe before triggering so the send succeeds
        let mut rx = state.round_trigger_tx.subscribe();
        let app = test_app(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/round/start")
                    .header("authorization", "Bearer my-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let resp: RoundStartResponse = serde_json::from_slice(&body).unwrap();
        assert!(resp.triggered);

        // Verify the trigger was received
        let received = rx.try_recv();
        assert!(received.is_ok());
    }

    #[tokio::test]
    async fn test_peers_endpoint() {
        let state = test_state("key");
        let app = test_app(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/peers")
                    .header("authorization", "Bearer key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let snapshot: PeerSnapshot = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.peers.len(), 1);
        assert_eq!(snapshot.peers[0].id, "peer1");
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = ApiRateLimiter::new(2); // 2 req/s, burst capacity 4
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // First 4 requests should be allowed (burst capacity = 2 * 2 = 4)
        assert!(limiter.check(ip));
        assert!(limiter.check(ip));
        assert!(limiter.check(ip));
        assert!(limiter.check(ip));
        // 5th should be rate limited (bucket exhausted)
        assert!(!limiter.check(ip));

        // Different IP should have its own bucket
        let ip2: IpAddr = "10.0.0.1".parse().unwrap();
        assert!(limiter.check(ip2));
    }
}
