//! JSON-RPC server for Helix node.
//!
//! Provides a JSON-RPC 2.0 interface over HTTP using axum.
//! Methods:
//! - `helix_getTrainingStatus` — current training round status
//! - `helix_getNetworkStatus` — network health and peer info
//! - `helix_submitTrainingStep` — trigger a training step (worker only)
//! - `helix_getProofStatus` — query proof submission status
//! - `helix_getWorkers` — list connected workers

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::api::http::OrchestratorSnapshot;

// ============================================================================
// JSON-RPC Protocol Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub id: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl JsonRpcResponse {
    fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    fn error(id: serde_json::Value, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(JsonRpcError { code, message }),
            id,
        }
    }

    fn method_not_found(id: serde_json::Value, method: &str) -> Self {
        Self::error(id, -32601, format!("Method not found: {}", method))
    }
}

// ============================================================================
// RPC Response Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingStatusResponse {
    pub current_round: Option<RoundStatusInfo>,
    pub completed_rounds: u64,
    pub is_training: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoundStatusInfo {
    pub round_id: u64,
    pub phase: String,
    pub gradients_received: usize,
    pub workers_assigned: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkStatusResponse {
    pub node_role: String,
    pub worker_count: usize,
    pub available_workers: usize,
    pub computing_workers: usize,
    pub uptime_secs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitStepResponse {
    pub accepted: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProofStatusResponse {
    pub round_id: u64,
    pub status: String,
    pub proofs_collected: usize,
    pub total_submitted: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerInfoResponse {
    pub id: String,
    pub status: String,
    pub rounds_completed: u64,
}

// ============================================================================
// RPC Server State
// ============================================================================

/// Shared state for the RPC server.
pub struct RpcState {
    /// Snapshot of orchestrator state (updated periodically by main loop).
    pub snapshot: Arc<RwLock<OrchestratorSnapshot>>,
    /// Channel to trigger a training round.
    pub round_trigger_tx: broadcast::Sender<()>,
    /// Node role ("worker" or "aggregator").
    pub node_role: String,
    /// Server start time for uptime calculation.
    pub start_time: std::time::Instant,
    /// Proof submission status: (round_id, status, proofs_collected).
    pub proof_status: Arc<RwLock<Vec<ProofStatusEntry>>>,
}

/// Tracks proof status per round.
#[derive(Debug, Clone)]
pub struct ProofStatusEntry {
    pub round_id: u64,
    pub status: String,
    pub proofs_collected: usize,
}

// ============================================================================
// RPC Handler
// ============================================================================

async fn rpc_handler(
    State(state): State<Arc<RpcState>>,
    Json(req): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    let response = match req.method.as_str() {
        "helix_getTrainingStatus" => handle_get_training_status(&state),
        "helix_getNetworkStatus" => handle_get_network_status(&state),
        "helix_submitTrainingStep" => handle_submit_training_step(&state),
        "helix_getProofStatus" => handle_get_proof_status(&state, &req.params),
        "helix_getWorkers" => handle_get_workers(&state),
        _ => JsonRpcResponse::method_not_found(req.id.clone(), &req.method),
    };

    // Preserve the request ID if we didn't already set it
    Json(JsonRpcResponse {
        id: req.id,
        ..response
    })
}

fn handle_get_training_status(state: &RpcState) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let current_round = snap.current_round.as_ref().map(|r| RoundStatusInfo {
        round_id: r.round_id,
        phase: r.phase.clone(),
        gradients_received: r.gradients_received,
        workers_assigned: r.workers_assigned,
    });

    let result = TrainingStatusResponse {
        is_training: current_round.is_some(),
        current_round,
        completed_rounds: snap.completed_rounds,
    };

    JsonRpcResponse::success(
        serde_json::Value::Null,
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_network_status(state: &RpcState) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let result = NetworkStatusResponse {
        node_role: state.node_role.clone(),
        worker_count: snap.worker_count,
        available_workers: snap.available_workers,
        computing_workers: snap.computing_workers,
        uptime_secs: state.start_time.elapsed().as_secs(),
    };

    JsonRpcResponse::success(
        serde_json::Value::Null,
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_submit_training_step(state: &RpcState) -> JsonRpcResponse {
    match state.round_trigger_tx.send(()) {
        Ok(_) => {
            let result = SubmitStepResponse {
                accepted: true,
                message: "Training step triggered".to_string(),
            };
            JsonRpcResponse::success(
                serde_json::Value::Null,
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        Err(_) => {
            let result = SubmitStepResponse {
                accepted: false,
                message: "No listeners for round trigger".to_string(),
            };
            JsonRpcResponse::success(
                serde_json::Value::Null,
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
    }
}

fn handle_get_proof_status(state: &RpcState, params: &serde_json::Value) -> JsonRpcResponse {
    let round_id = params
        .get("round_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let proof_status = state.proof_status.read();
    let entry = proof_status.iter().find(|e| e.round_id == round_id);

    let result = match entry {
        Some(e) => ProofStatusResponse {
            round_id: e.round_id,
            status: e.status.clone(),
            proofs_collected: e.proofs_collected,
            total_submitted: proof_status.len() as u64,
        },
        None => ProofStatusResponse {
            round_id,
            status: "unknown".to_string(),
            proofs_collected: 0,
            total_submitted: proof_status.len() as u64,
        },
    };

    JsonRpcResponse::success(
        serde_json::Value::Null,
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_workers(state: &RpcState) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let workers: Vec<WorkerInfoResponse> = snap
        .workers
        .iter()
        .map(|w| WorkerInfoResponse {
            id: w.id.clone(),
            status: w.status.clone(),
            rounds_completed: w.rounds_completed,
        })
        .collect();

    JsonRpcResponse::success(
        serde_json::Value::Null,
        serde_json::to_value(workers).unwrap_or_default(),
    )
}

// ============================================================================
// Server Startup
// ============================================================================

/// Starts the JSON-RPC server on the given address.
pub async fn start_rpc_server(
    addr: SocketAddr,
    state: Arc<RpcState>,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/rpc", post(rpc_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("JSON-RPC server listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> Arc<RpcState> {
        let (tx, _) = broadcast::channel(16);
        Arc::new(RpcState {
            snapshot: Arc::new(RwLock::new(OrchestratorSnapshot {
                worker_count: 3,
                available_workers: 2,
                computing_workers: 1,
                completed_rounds: 5,
                ..Default::default()
            })),
            round_trigger_tx: tx,
            node_role: "aggregator".to_string(),
            start_time: std::time::Instant::now(),
            proof_status: Arc::new(RwLock::new(vec![
                ProofStatusEntry {
                    round_id: 1,
                    status: "committed".to_string(),
                    proofs_collected: 3,
                },
            ])),
        })
    }

    #[test]
    fn test_get_training_status() {
        let state = create_test_state();
        let response = handle_get_training_status(&state);
        assert!(response.result.is_some());

        let result: TrainingStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.completed_rounds, 5);
        assert!(!result.is_training);
    }

    #[test]
    fn test_get_network_status() {
        let state = create_test_state();
        let response = handle_get_network_status(&state);
        assert!(response.result.is_some());

        let result: NetworkStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.node_role, "aggregator");
        assert_eq!(result.worker_count, 3);
        assert_eq!(result.available_workers, 2);
    }

    #[test]
    fn test_get_proof_status_found() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 1});
        let response = handle_get_proof_status(&state, &params);
        assert!(response.result.is_some());

        let result: ProofStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.round_id, 1);
        assert_eq!(result.status, "committed");
        assert_eq!(result.proofs_collected, 3);
    }

    #[test]
    fn test_get_proof_status_not_found() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 999});
        let response = handle_get_proof_status(&state, &params);
        assert!(response.result.is_some());

        let result: ProofStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.status, "unknown");
    }

    #[test]
    fn test_get_workers() {
        let state = create_test_state();
        let response = handle_get_workers(&state);
        assert!(response.result.is_some());

        let result: Vec<WorkerInfoResponse> =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.is_empty()); // No workers in default snapshot
    }

    #[test]
    fn test_submit_training_step() {
        let state = create_test_state();
        // Need a receiver to avoid send error
        let mut _rx = state.round_trigger_tx.subscribe();
        let response = handle_submit_training_step(&state);
        assert!(response.result.is_some());

        let result: SubmitStepResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.accepted);
    }

    #[test]
    fn test_method_not_found() {
        let response = JsonRpcResponse::method_not_found(
            serde_json::Value::Number(1.into()),
            "nonexistent_method",
        );
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_json_rpc_response_serialization() {
        let response = JsonRpcResponse::success(
            serde_json::Value::Number(1.into()),
            serde_json::json!({"status": "ok"}),
        );
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"status\":\"ok\""));
        // error field should be omitted
        assert!(!json.contains("\"error\""));
    }
}
