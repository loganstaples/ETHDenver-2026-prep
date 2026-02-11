//! JSON-RPC server for Helix node.
//!
//! Provides a JSON-RPC 2.0 interface over HTTP using axum.
//! Methods:
//! - `helix_getTrainingStatus` — current training round status
//! - `helix_getNetworkStatus` — network health and peer info
//! - `helix_submitTrainingStep` — trigger a training step (worker only)
//! - `helix_getProofStatus` — query proof submission status
//! - `helix_getWorkers` — list connected workers
//! - `helix_submitProof` — submit a ZK proof for a training round
//! - `helix_getModelState` — query on-chain model state
//! - `helix_startTraining` — start a new training round
//! - `helix_stopTraining` — stop the current training session
//! - `helix_getConfig` — get node configuration
//! - `helix_getAggregationResult` — get aggregation result for a round
//! - `helix_getMPCStatus` — get MPC training status

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

    fn invalid_params(id: serde_json::Value, msg: &str) -> Self {
        Self::error(id, -32602, format!("Invalid params: {}", msg))
    }

    fn internal_error(id: serde_json::Value, msg: &str) -> Self {
        Self::error(id, -32603, format!("Internal error: {}", msg))
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

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitProofResponse {
    pub accepted: bool,
    pub proof_hash: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelStateResponse {
    pub model_id: u64,
    pub current_round: u64,
    pub current_commitment: String,
    pub active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartTrainingResponse {
    pub started: bool,
    pub session_id: Option<u64>,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StopTrainingResponse {
    pub stopped: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeConfigResponse {
    pub node_role: String,
    pub model_id: Option<u64>,
    pub rpc_endpoint: String,
    pub mpc_enabled: bool,
    pub mpc_num_parties: usize,
    pub verification_enabled: bool,
    pub max_workers: usize,
    pub round_timeout_secs: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregationResultResponse {
    pub round_id: u64,
    pub proofs_aggregated: usize,
    pub pedersen_aggregate: String,
    pub merkle_root: String,
    pub status: String,
    pub submitted_on_chain: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MPCStatusResponse {
    pub enabled: bool,
    pub num_parties: usize,
    pub party_index: Option<usize>,
    pub current_step: u64,
    pub reshare_interval: u64,
    pub verify_commitments: bool,
    pub model_dims: MPCModelDimsResponse,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MPCModelDimsResponse {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
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
    /// Channel to stop training.
    pub stop_trigger_tx: broadcast::Sender<()>,
    /// Node role ("worker" or "aggregator").
    pub node_role: String,
    /// Server start time for uptime calculation.
    pub start_time: std::time::Instant,
    /// Proof submission status: (round_id, status, proofs_collected).
    pub proof_status: Arc<RwLock<Vec<ProofStatusEntry>>>,
    /// Submitted proofs queue: (model_id, round_id, proof_bytes, public_inputs_hex).
    pub proof_queue: Arc<RwLock<Vec<QueuedProof>>>,
    /// Node configuration.
    pub node_config: Arc<RwLock<NodeConfigSnapshot>>,
    /// Aggregation results per round.
    pub aggregation_results: Arc<RwLock<Vec<AggregationResultEntry>>>,
    /// MPC status.
    pub mpc_status: Arc<RwLock<MPCStatusSnapshot>>,
    /// Active model ID.
    pub model_id: Arc<RwLock<Option<u64>>>,
    /// RPC listen address (for config reporting).
    pub rpc_addr: String,
}

/// Tracks proof status per round.
#[derive(Debug, Clone)]
pub struct ProofStatusEntry {
    pub round_id: u64,
    pub status: String,
    pub proofs_collected: usize,
}

/// A proof queued for on-chain submission.
#[derive(Debug, Clone)]
pub struct QueuedProof {
    pub model_id: u64,
    pub round_id: u64,
    pub proof_hex: String,
    pub public_inputs_hex: Vec<String>,
    pub submitted: bool,
    pub tx_hash: Option<String>,
}

/// Snapshot of node configuration.
#[derive(Debug, Clone)]
pub struct NodeConfigSnapshot {
    pub max_workers: usize,
    pub round_timeout_secs: u64,
    pub verification_enabled: bool,
}

impl Default for NodeConfigSnapshot {
    fn default() -> Self {
        Self {
            max_workers: 10,
            round_timeout_secs: 300,
            verification_enabled: true,
        }
    }
}

/// Snapshot of aggregation result for a round.
#[derive(Debug, Clone)]
pub struct AggregationResultEntry {
    pub round_id: u64,
    pub proofs_aggregated: usize,
    pub pedersen_aggregate: [u8; 32],
    pub merkle_root: [u8; 32],
    pub status: String,
    pub submitted_on_chain: bool,
}

/// Snapshot of MPC status.
#[derive(Debug, Clone)]
pub struct MPCStatusSnapshot {
    pub enabled: bool,
    pub num_parties: usize,
    pub party_index: Option<usize>,
    pub current_step: u64,
    pub reshare_interval: u64,
    pub verify_commitments: bool,
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
}

impl Default for MPCStatusSnapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            num_parties: 3,
            party_index: None,
            current_step: 0,
            reshare_interval: 50,
            verify_commitments: true,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
        }
    }
}

// ============================================================================
// RPC Handler
// ============================================================================

async fn rpc_handler(
    State(state): State<Arc<RpcState>>,
    Json(req): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    let id = req.id.clone();
    let response = match req.method.as_str() {
        "helix_getTrainingStatus" => handle_get_training_status(&state, &id),
        "helix_getNetworkStatus" => handle_get_network_status(&state, &id),
        "helix_submitTrainingStep" => handle_submit_training_step(&state, &id),
        "helix_getProofStatus" => handle_get_proof_status(&state, &req.params, &id),
        "helix_getWorkers" => handle_get_workers(&state, &id),
        "helix_submitProof" => handle_submit_proof(&state, &req.params, &id),
        "helix_getModelState" => handle_get_model_state(&state, &req.params, &id),
        "helix_startTraining" => handle_start_training(&state, &req.params, &id),
        "helix_stopTraining" => handle_stop_training(&state, &id),
        "helix_getConfig" => handle_get_config(&state, &id),
        "helix_getAggregationResult" => handle_get_aggregation_result(&state, &req.params, &id),
        "helix_getMPCStatus" => handle_get_mpc_status(&state, &id),
        "helix_getTrainingResult" => handle_get_training_result(&state, &req.params, &id),
        "helix_generateProof" => handle_generate_proof(&state, &req.params, &id),
        "helix_registerModel" => handle_register_model(&state, &req.params, &id),
        "helix_stake" => handle_stake(&state, &req.params, &id),
        "helix_unstake" => handle_unstake(&state, &req.params, &id),
        _ => JsonRpcResponse::method_not_found(id.clone(), &req.method),
    };

    // Preserve the request ID
    Json(JsonRpcResponse {
        id,
        ..response
    })
}

fn handle_get_training_status(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
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
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_network_status(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let result = NetworkStatusResponse {
        node_role: state.node_role.clone(),
        worker_count: snap.worker_count,
        available_workers: snap.available_workers,
        computing_workers: snap.computing_workers,
        uptime_secs: state.start_time.elapsed().as_secs(),
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_submit_training_step(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    match state.round_trigger_tx.send(()) {
        Ok(_) => {
            let result = SubmitStepResponse {
                accepted: true,
                message: "Training step triggered".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        Err(_) => {
            let result = SubmitStepResponse {
                accepted: false,
                message: "No listeners for round trigger".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
    }
}

fn handle_get_proof_status(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
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
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_workers(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
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
        id.clone(),
        serde_json::to_value(workers).unwrap_or_default(),
    )
}

fn handle_submit_proof(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = match params.get("model_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "model_id is required (u64)"),
    };
    let round_id = match params.get("round_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "round_id is required (u64)"),
    };
    let proof_hex = match params.get("proof").and_then(|v| v.as_str()) {
        Some(v) => v.to_string(),
        None => return JsonRpcResponse::invalid_params(id.clone(), "proof is required (hex string)"),
    };
    let public_inputs: Vec<String> = match params.get("public_inputs") {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        }
        _ => return JsonRpcResponse::invalid_params(id.clone(), "public_inputs is required (array of hex strings)"),
    };

    // Validate hex encoding
    let proof_stripped = proof_hex.strip_prefix("0x").unwrap_or(&proof_hex);
    if hex::decode(proof_stripped).is_err() {
        return JsonRpcResponse::invalid_params(id.clone(), "proof is not valid hex");
    }

    // Compute a hash for tracking
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(proof_stripped.as_bytes());
    let proof_hash = hex::encode(&hasher.finalize()[..16]);

    let queued = QueuedProof {
        model_id,
        round_id,
        proof_hex,
        public_inputs_hex: public_inputs,
        submitted: false,
        tx_hash: None,
    };

    state.proof_queue.write().push(queued);

    let result = SubmitProofResponse {
        accepted: true,
        proof_hash: format!("0x{}", proof_hash),
        message: format!("Proof queued for model {} round {}", model_id, round_id),
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_model_state(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = match params.get("model_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => {
            // Fall back to active model ID
            match *state.model_id.read() {
                Some(mid) => mid,
                None => return JsonRpcResponse::invalid_params(id.clone(), "model_id is required (u64)"),
            }
        }
    };

    // Return locally cached model state from the snapshot
    let snap = state.snapshot.read();
    let current_round = snap.current_round.as_ref().map(|r| r.round_id).unwrap_or(0);

    // Use real commitment from round state if available
    let current_commitment = snap
        .current_round
        .as_ref()
        .and_then(|r| r.commitment_hash.clone())
        .unwrap_or_else(|| {
            // Check aggregation results for the latest committed round
            let agg = state.aggregation_results.read();
            if let Some(latest) = agg.last() {
                format!("0x{}", hex::encode(latest.merkle_root))
            } else {
                // No data available — return null instead of fake zeros
                "null".to_string()
            }
        });

    let result = ModelStateResponse {
        model_id,
        current_round,
        current_commitment,
        active: snap.current_round.is_some() || snap.completed_rounds > 0,
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_start_training(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    // Check if already training
    {
        let snap = state.snapshot.read();
        if snap.current_round.is_some() {
            return JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(StartTrainingResponse {
                    started: false,
                    session_id: None,
                    message: "Training already in progress".to_string(),
                }).unwrap_or_default(),
            );
        }
    }

    // Store model_id if provided
    if let Some(mid) = params.get("model_id").and_then(|v| v.as_u64()) {
        *state.model_id.write() = Some(mid);
    }

    // Trigger round start
    match state.round_trigger_tx.send(()) {
        Ok(_) => {
            let result = StartTrainingResponse {
                started: true,
                session_id: Some(1), // Assigned by orchestrator
                message: "Training round initiated".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        Err(_) => {
            let result = StartTrainingResponse {
                started: false,
                session_id: None,
                message: "No training orchestrator listening".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
    }
}

fn handle_stop_training(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    match state.stop_trigger_tx.send(()) {
        Ok(_) => {
            let result = StopTrainingResponse {
                stopped: true,
                message: "Training stop signal sent".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        Err(_) => {
            let result = StopTrainingResponse {
                stopped: false,
                message: "No training session active or no listeners".to_string(),
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
    }
}

fn handle_get_config(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let config = state.node_config.read();
    let mpc = state.mpc_status.read();
    let model_id = *state.model_id.read();

    let result = NodeConfigResponse {
        node_role: state.node_role.clone(),
        model_id,
        rpc_endpoint: state.rpc_addr.clone(),
        mpc_enabled: mpc.enabled,
        mpc_num_parties: mpc.num_parties,
        verification_enabled: config.verification_enabled,
        max_workers: config.max_workers,
        round_timeout_secs: config.round_timeout_secs,
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_get_aggregation_result(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let round_id = match params.get("round_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "round_id is required (u64)"),
    };

    let results = state.aggregation_results.read();
    let entry = results.iter().find(|e| e.round_id == round_id);

    match entry {
        Some(e) => {
            let result = AggregationResultResponse {
                round_id: e.round_id,
                proofs_aggregated: e.proofs_aggregated,
                pedersen_aggregate: format!("0x{}", hex::encode(e.pedersen_aggregate)),
                merkle_root: format!("0x{}", hex::encode(e.merkle_root)),
                status: e.status.clone(),
                submitted_on_chain: e.submitted_on_chain,
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32000,
            format!("No aggregation result for round {}", round_id),
        ),
    }
}

fn handle_get_mpc_status(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let mpc = state.mpc_status.read();

    let result = MPCStatusResponse {
        enabled: mpc.enabled,
        num_parties: mpc.num_parties,
        party_index: mpc.party_index,
        current_step: mpc.current_step,
        reshare_interval: mpc.reshare_interval,
        verify_commitments: mpc.verify_commitments,
        model_dims: MPCModelDimsResponse {
            d_in: mpc.d_in,
            d_hid: mpc.d_hid,
            d_out: mpc.d_out,
        },
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

/// Response for `helix_generateProof`.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateProofResponse {
    pub accepted: bool,
    pub message: String,
    pub round_id: u64,
}

/// Response for `helix_registerModel`.
#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterModelResponse {
    pub registered: bool,
    pub model_id: u64,
    pub message: String,
}

/// Response for `helix_stake` / `helix_unstake`.
#[derive(Debug, Serialize, Deserialize)]
pub struct StakeResponse {
    pub success: bool,
    pub message: String,
}

/// Response for `helix_getTrainingResult`.
///
/// Returns the latest training round result including the aggregated weight
/// update data needed for external proof generation.
#[derive(Debug, Serialize)]
struct TrainingResultResponse {
    /// Whether a completed round result is available.
    available: bool,
    /// Latest completed round ID.
    round_id: u64,
    /// Number of workers that contributed.
    worker_count: u64,
    /// Loss value after this round.
    loss: f64,
    /// Accumulated error bound.
    error_bound: f64,
    /// Model dimensions (d_in, d_hid, d_out).
    model_dims: Option<(usize, usize, usize)>,
    /// Current step number.
    step_number: u64,
}

fn handle_get_training_result(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let snap = state.snapshot.read();

    let round_id = params
        .get("round_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(snap.completed_rounds);

    // Check if we have an aggregation result for this round
    let agg_results = state.aggregation_results.read();
    let has_agg = agg_results.iter().any(|r| r.round_id == round_id);

    let result = TrainingResultResponse {
        available: has_agg || snap.completed_rounds > 0,
        round_id,
        worker_count: snap.available_workers as u64,
        loss: 0.0, // Loss is tracked externally by the prover
        error_bound: 0.0,
        model_dims: None,
        step_number: round_id,
    };

    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_generate_proof(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let round_id = params
        .get("round_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Trigger a training step which includes proof generation
    match state.round_trigger_tx.send(()) {
        Ok(_) => {
            let result = GenerateProofResponse {
                accepted: true,
                message: format!("Proof generation triggered for round {}", round_id),
                round_id,
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
        Err(_) => {
            let result = GenerateProofResponse {
                accepted: false,
                message: "No training orchestrator listening for proof generation".to_string(),
                round_id,
            };
            JsonRpcResponse::success(
                id.clone(),
                serde_json::to_value(result).unwrap_or_default(),
            )
        }
    }
}

fn handle_register_model(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);

    // Store the model_id in node state
    *state.model_id.write() = Some(model_id);

    let result = RegisterModelResponse {
        registered: true,
        model_id,
        message: format!("Model {} registered in node state", model_id),
    };
    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_stake(
    _state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = match params.get("model_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "model_id is required (u64)"),
    };

    // Staking is handled on-chain; node acknowledges the intent
    let result = StakeResponse {
        success: true,
        message: format!(
            "Stake request acknowledged for model {}. On-chain staking must be performed via the coordinator contract.",
            model_id
        ),
    };
    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
    )
}

fn handle_unstake(
    _state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = match params.get("model_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "model_id is required (u64)"),
    };

    let result = StakeResponse {
        success: true,
        message: format!(
            "Unstake request acknowledged for model {}. On-chain unstaking must be performed via the coordinator contract.",
            model_id
        ),
    };
    JsonRpcResponse::success(
        id.clone(),
        serde_json::to_value(result).unwrap_or_default(),
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

/// Creates a default RpcState for testing or simple setups.
pub fn create_default_rpc_state(
    node_role: &str,
    rpc_addr: &str,
) -> Arc<RpcState> {
    let (round_tx, _) = broadcast::channel(16);
    let (stop_tx, _) = broadcast::channel(16);
    Arc::new(RpcState {
        snapshot: Arc::new(RwLock::new(OrchestratorSnapshot::default())),
        round_trigger_tx: round_tx,
        stop_trigger_tx: stop_tx,
        node_role: node_role.to_string(),
        start_time: std::time::Instant::now(),
        proof_status: Arc::new(RwLock::new(Vec::new())),
        proof_queue: Arc::new(RwLock::new(Vec::new())),
        node_config: Arc::new(RwLock::new(NodeConfigSnapshot::default())),
        aggregation_results: Arc::new(RwLock::new(Vec::new())),
        mpc_status: Arc::new(RwLock::new(MPCStatusSnapshot::default())),
        model_id: Arc::new(RwLock::new(None)),
        rpc_addr: rpc_addr.to_string(),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> Arc<RpcState> {
        let (tx, _) = broadcast::channel(16);
        let (stop_tx, _) = broadcast::channel(16);
        Arc::new(RpcState {
            snapshot: Arc::new(RwLock::new(OrchestratorSnapshot {
                worker_count: 3,
                available_workers: 2,
                computing_workers: 1,
                completed_rounds: 5,
                ..Default::default()
            })),
            round_trigger_tx: tx,
            stop_trigger_tx: stop_tx,
            node_role: "aggregator".to_string(),
            start_time: std::time::Instant::now(),
            proof_status: Arc::new(RwLock::new(vec![
                ProofStatusEntry {
                    round_id: 1,
                    status: "committed".to_string(),
                    proofs_collected: 3,
                },
            ])),
            proof_queue: Arc::new(RwLock::new(Vec::new())),
            node_config: Arc::new(RwLock::new(NodeConfigSnapshot::default())),
            aggregation_results: Arc::new(RwLock::new(vec![
                AggregationResultEntry {
                    round_id: 1,
                    proofs_aggregated: 3,
                    pedersen_aggregate: [0xab; 32],
                    merkle_root: [0xcd; 32],
                    status: "committed".to_string(),
                    submitted_on_chain: true,
                },
            ])),
            mpc_status: Arc::new(RwLock::new(MPCStatusSnapshot {
                enabled: true,
                num_parties: 3,
                party_index: Some(0),
                current_step: 42,
                ..Default::default()
            })),
            model_id: Arc::new(RwLock::new(Some(1))),
            rpc_addr: "127.0.0.1:9545".to_string(),
        })
    }

    fn null_id() -> serde_json::Value {
        serde_json::Value::Number(1.into())
    }

    #[test]
    fn test_get_training_status() {
        let state = create_test_state();
        let response = handle_get_training_status(&state, &null_id());
        assert!(response.result.is_some());

        let result: TrainingStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.completed_rounds, 5);
        assert!(!result.is_training);
    }

    #[test]
    fn test_get_network_status() {
        let state = create_test_state();
        let response = handle_get_network_status(&state, &null_id());
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
        let response = handle_get_proof_status(&state, &params, &null_id());
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
        let response = handle_get_proof_status(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: ProofStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.status, "unknown");
    }

    #[test]
    fn test_get_workers() {
        let state = create_test_state();
        let response = handle_get_workers(&state, &null_id());
        assert!(response.result.is_some());

        let result: Vec<WorkerInfoResponse> =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.is_empty()); // No workers in default snapshot
    }

    #[test]
    fn test_submit_training_step() {
        let state = create_test_state();
        let mut _rx = state.round_trigger_tx.subscribe();
        let response = handle_submit_training_step(&state, &null_id());
        assert!(response.result.is_some());

        let result: SubmitStepResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.accepted);
    }

    #[test]
    fn test_submit_proof() {
        let state = create_test_state();
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "0xdeadbeef",
            "public_inputs": ["0x01", "0x02", "0x03", "0x04", "0x05", "0x06", "0x07"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: SubmitProofResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.accepted);
        assert!(result.proof_hash.starts_with("0x"));

        // Verify proof was queued
        let queue = state.proof_queue.read();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].model_id, 1);
        assert_eq!(queue[0].round_id, 1);
    }

    #[test]
    fn test_submit_proof_invalid_hex() {
        let state = create_test_state();
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "not-hex-data!!",
            "public_inputs": ["0x01"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }

    #[test]
    fn test_submit_proof_missing_params() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.error.is_some());
    }

    #[test]
    fn test_get_model_state() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_get_model_state(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: ModelStateResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.model_id, 1);
        assert!(result.active);
    }

    #[test]
    fn test_get_model_state_uses_active_model() {
        let state = create_test_state();
        let params = serde_json::json!({});
        let response = handle_get_model_state(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: ModelStateResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.model_id, 1); // Uses the active model_id from state
    }

    #[test]
    fn test_start_training() {
        let state = create_test_state();
        let mut _rx = state.round_trigger_tx.subscribe();
        let params = serde_json::json!({"model_id": 2});
        let response = handle_start_training(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: StartTrainingResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.started);

        // Verify model_id was updated
        assert_eq!(*state.model_id.read(), Some(2));
    }

    #[test]
    fn test_stop_training() {
        let state = create_test_state();
        let mut _rx = state.stop_trigger_tx.subscribe();
        let response = handle_stop_training(&state, &null_id());
        assert!(response.result.is_some());

        let result: StopTrainingResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.stopped);
    }

    #[test]
    fn test_get_config() {
        let state = create_test_state();
        let response = handle_get_config(&state, &null_id());
        assert!(response.result.is_some());

        let result: NodeConfigResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.node_role, "aggregator");
        assert!(result.mpc_enabled);
        assert_eq!(result.mpc_num_parties, 3);
        assert_eq!(result.max_workers, 10);
        assert!(result.verification_enabled);
        assert_eq!(result.rpc_endpoint, "127.0.0.1:9545");
    }

    #[test]
    fn test_get_aggregation_result() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 1});
        let response = handle_get_aggregation_result(&state, &params, &null_id());
        assert!(response.result.is_some());

        let result: AggregationResultResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert_eq!(result.round_id, 1);
        assert_eq!(result.proofs_aggregated, 3);
        assert!(result.submitted_on_chain);
        assert!(result.pedersen_aggregate.starts_with("0x"));
        assert!(result.merkle_root.starts_with("0x"));
    }

    #[test]
    fn test_get_aggregation_result_not_found() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 999});
        let response = handle_get_aggregation_result(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32000);
    }

    #[test]
    fn test_get_mpc_status() {
        let state = create_test_state();
        let response = handle_get_mpc_status(&state, &null_id());
        assert!(response.result.is_some());

        let result: MPCStatusResponse =
            serde_json::from_value(response.result.unwrap()).unwrap();
        assert!(result.enabled);
        assert_eq!(result.num_parties, 3);
        assert_eq!(result.party_index, Some(0));
        assert_eq!(result.current_step, 42);
        assert_eq!(result.model_dims.d_in, 4);
        assert_eq!(result.model_dims.d_hid, 8);
        assert_eq!(result.model_dims.d_out, 2);
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

    #[test]
    fn test_create_default_rpc_state() {
        let state = create_default_rpc_state("worker", "0.0.0.0:9545");
        assert_eq!(state.node_role, "worker");
        assert_eq!(state.rpc_addr, "0.0.0.0:9545");
        assert!(state.model_id.read().is_none());
    }
}
