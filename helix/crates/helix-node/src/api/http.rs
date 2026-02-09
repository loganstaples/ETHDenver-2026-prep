//! Minimal HTTP API for aggregator node monitoring and control.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::round_commit::RoundCommitManager;
use crate::training::orchestrator::{RoundPhase, TrainingOrchestrator};

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
}

/// Snapshot of orchestrator state (updated periodically).
#[derive(Debug, Clone, Default, Serialize)]
pub struct OrchestratorSnapshot {
    /// Number of connected workers.
    pub worker_count: usize,
    /// Number of available workers.
    pub available_workers: usize,
    /// Number of computing workers.
    pub computing_workers: usize,
    /// Current round info.
    pub current_round: Option<RoundInfo>,
    /// Completed round count.
    pub completed_rounds: u64,
    /// Worker list.
    pub workers: Vec<WorkerInfo>,
}

/// Round info for API response.
#[derive(Debug, Clone, Serialize)]
pub struct RoundInfo {
    pub round_id: u64,
    pub phase: String,
    pub gradients_received: usize,
    pub workers_assigned: usize,
}

/// Worker info for API response.
#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
}

/// A single peer entry for the peers endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerEntry {
    pub id: String,
    pub address: String,
    pub last_seen: u64,
    pub reputation: i64,
}

/// Snapshot of connected peers (returned by `/peers`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PeerSnapshot {
    pub peers: Vec<PeerEntry>,
}

/// Aggregate node metrics (returned by `/metrics`).
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

/// Health response.
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    role: String,
    workers: usize,
    current_round: Option<u64>,
}

/// Round start response.
#[derive(Serialize)]
struct RoundStartResponse {
    triggered: bool,
    message: String,
}

/// Starts the HTTP API server on the given address.
pub async fn start_api_server(
    addr: SocketAddr,
    state: Arc<ApiState>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/round/start", post(round_start_handler))
        .route("/round/status", get(round_status_handler))
        .route("/workers", get(workers_handler))
        .route("/peers", get(peers_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("HTTP API listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

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
