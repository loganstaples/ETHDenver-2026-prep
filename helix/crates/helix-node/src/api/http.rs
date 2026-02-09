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

use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

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
    /// Bearer token required for authenticated endpoints.
    /// Must match the `Authorization: Bearer <token>` header.
    pub api_key: String,
    /// Per-IP rate limiter.
    pub rate_limiter: Arc<ApiRateLimiter>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Snapshot of orchestrator state (updated periodically).
#[derive(Debug, Clone, Default, Serialize)]
pub struct OrchestratorSnapshot {
    pub worker_count: usize,
    pub available_workers: usize,
    pub computing_workers: usize,
    pub current_round: Option<RoundInfo>,
    pub completed_rounds: u64,
    pub workers: Vec<WorkerInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundInfo {
    pub round_id: u64,
    pub phase: String,
    pub gradients_received: usize,
    pub workers_assigned: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
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

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    role: String,
    workers: usize,
    current_round: Option<u64>,
}

#[derive(Serialize)]
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
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Combine and add rate limiting to everything
    let app = Router::new()
        .merge(public_routes)
        .merge(auth_routes)
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("HTTP API listening on {}", addr);

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
    Json(HealthResponse {
        status: "healthy".to_string(),
        role: "aggregator".to_string(),
        workers: snapshot.worker_count,
        current_round: snapshot.current_round.as_ref().map(|r| r.round_id),
    })
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

async fn metrics_handler(State(state): State<Arc<ApiState>>) -> Json<MetricsSnapshot> {
    let snapshot = state.metrics.read().clone();
    Json(snapshot)
}
