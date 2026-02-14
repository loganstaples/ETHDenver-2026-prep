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
// RPC Response Types (node-internal, kept for backward compat with node tests)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitStepResponse {
    pub accepted: bool,
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
// Helper: map node worker status strings to client WorkerStatus variant names
// ============================================================================

fn map_worker_status(status: &str) -> &'static str {
    match status.to_lowercase().as_str() {
        "idle" | "available" | "ready" => "Idle",
        "training" | "computing" | "busy" => "Training",
        "proving" | "generating_proof" => "Proving",
        "offline" | "disconnected" => "Offline",
        "connecting" | "joined" => "Connecting",
        "syncing" | "synchronizing" => "Syncing",
        "waiting" | "pending" => "Waiting",
        "faulted" | "error" | "failed" => "Faulted",
        "slashed" => "Slashed",
        _ => "Idle",
    }
}

// ============================================================================
// RPC Server State
// ============================================================================

/// Maximum proof size in bytes (10 MB). Proofs larger than this are rejected.
const MAX_PROOF_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of proofs in the submission queue before rejecting new ones.
const MAX_PROOF_QUEUE_SIZE: usize = 1000;

/// Maximum number of public inputs per proof submission.
const MAX_PUBLIC_INPUTS: usize = 64;

/// RPC-layer rate limiter for proof submissions.
///
/// Uses a simple sliding-window token bucket: each caller gets `burst` tokens
/// that refill at `rate_per_sec` tokens/second. Proof submissions cost 1 token.
pub struct RpcRateLimiter {
    /// Per-caller token state: (tokens_remaining, last_refill_time).
    callers: std::collections::HashMap<String, (f64, std::time::Instant)>,
    /// Tokens per second refill rate.
    pub rate_per_sec: f64,
    /// Maximum burst size (bucket capacity).
    pub burst: f64,
}

impl RpcRateLimiter {
    /// Creates a new RPC rate limiter.
    pub fn new(rate_per_sec: f64, burst: f64) -> Self {
        Self {
            callers: std::collections::HashMap::new(),
            rate_per_sec,
            burst,
        }
    }

    /// Checks if a caller is allowed to submit. Returns true if allowed.
    pub fn check(&mut self, caller_id: &str) -> bool {
        let now = std::time::Instant::now();
        let (tokens, last) = self
            .callers
            .entry(caller_id.to_string())
            .or_insert((self.burst, now));

        // Refill tokens
        let elapsed = now.duration_since(*last).as_secs_f64();
        *tokens = (*tokens + elapsed * self.rate_per_sec).min(self.burst);
        *last = now;

        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Number of tracked callers.
    pub fn caller_count(&self) -> usize {
        self.callers.len()
    }
}

impl Default for RpcRateLimiter {
    fn default() -> Self {
        // 10 proof submissions per second, burst of 20
        Self::new(10.0, 20.0)
    }
}

/// Completed round weight metadata for result distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundWeightEntry {
    /// Round number.
    pub round_id: u64,
    /// Model ID.
    pub model_id: u64,
    /// SHA-256 commitment hash of the weight bytes.
    pub commitment: [u8; 32],
    /// Serialized weight bytes (the actual model weights).
    pub weight_bytes: Vec<u8>,
    /// Final loss value for this round.
    pub loss: f64,
    /// Accumulated error bound.
    pub error_bound: f64,
    /// Number of training steps completed in this round.
    pub steps_completed: u64,
    /// Number of workers that contributed.
    pub num_contributors: u32,
    /// Completion timestamp (unix seconds).
    pub completed_at: u64,
    /// On-chain transaction hash (hex), if submitted.
    pub tx_hash: Option<String>,
}

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
    /// Rate limiter for proof submissions.
    pub rate_limiter: Arc<RwLock<RpcRateLimiter>>,
    /// Completed round weights for result distribution.
    pub round_weights: Arc<RwLock<Vec<RoundWeightEntry>>>,
    /// Worker daemon state (for worker status, earnings, auto-join RPC).
    pub worker_daemon: Option<Arc<crate::worker::WorkerDaemon>>,
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
        "helix_getNetworkStatus" | "helix_networkStatus" => handle_get_network_status(&state, &id),
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
        // New handlers for client compatibility
        "helix_health" => handle_health(&state, &id),
        "helix_capabilities" => handle_capabilities(&state, &id),
        "helix_getTrainingProgress" => handle_get_training_progress(&state, &id),
        "helix_verifyProof" => handle_verify_proof(&state, &id),
        "helix_listModels" => handle_list_models(&state, &id),
        "helix_getCurrentRound" => handle_get_current_round(&state, &id),
        "helix_getRound" => handle_get_round(&state, &req.params, &id),
        "helix_getWorker" => handle_get_worker(&state, &req.params, &id),
        "helix_getSelfWorker" => handle_get_self_worker(&state, &id),
        "helix_getStakingInfo" => handle_get_staking_info(&state, &id),
        "helix_claimRewards" => handle_claim_rewards(&state, &id),
        // Result distribution handlers
        "helix_getModelWeights" => handle_get_model_weights(&state, &req.params, &id),
        "helix_getTrainingHistory" => handle_get_training_history(&state, &req.params, &id),
        "helix_getTrainingReport" => handle_get_training_report(&state, &req.params, &id),
        "helix_downloadModel" => handle_download_model(&state, &req.params, &id),
        // Worker daemon handlers
        "helix_getWorkerStatus" => handle_get_worker_status(&state, &id),
        "helix_getEarnings" => handle_get_earnings(&state, &id),
        "helix_setAutoJoin" => handle_set_auto_join(&state, &req.params, &id),
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
    let is_training = snap.current_round.is_some();
    let current_round_id = snap.current_round.as_ref().map(|r| r.round_id).unwrap_or(0);
    let model_id = state.model_id.read().unwrap_or(0);

    // Map node round phase to client TrainingPhase variant name
    let phase = snap.current_round.as_ref()
        .map(|r| match r.phase.to_lowercase().as_str() {
            "forward" => "Forward",
            "backward" => "Backward",
            "gradient" | "gradient_compute" => "GradientCompute",
            "proof" | "proving" | "proof_generation" => "ProofGeneration",
            "submission" | "proof_submission" => "ProofSubmission",
            "verification" | "proof_verification" => "ProofVerification",
            "weight_update" | "update" => "WeightUpdate",
            "checkpoint" | "checkpointing" => "Checkpointing",
            "complete" | "round_complete" => "RoundComplete",
            "initializing" | "init" => "Initializing",
            "loading" | "loading_data" => "LoadingData",
            "failed" | "error" => "Failed",
            _ => if is_training { "Forward" } else { "Idle" },
        })
        .unwrap_or(if is_training { "Initializing" } else { "Idle" });

    let result = serde_json::json!({
        "active": is_training,
        "phase": phase,
        "current_round": current_round_id,
        "total_rounds": snap.completed_rounds + if is_training { 1 } else { 0 },
        "current_loss": 0.0,
        "accumulated_error": 0.0,
        "max_error_bound": 1000.0,
        "round_elapsed_ms": 0_u64,
        "estimated_remaining_ms": 0_u64,
        "model_id": model_id,
        "started_at": 0_i64
    });

    JsonRpcResponse::success(id.clone(), result)
}

fn handle_get_network_status(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let result = serde_json::json!({
        "peer_count": snap.worker_count as u32,
        "active_workers": snap.available_workers as u32,
        "active_aggregators": if state.node_role == "aggregator" { 1_u32 } else { 0_u32 },
        "avg_latency_ms": 0_u64,
        "block_height": 0_u64,
        "chain_id": 31337_u64,
        "blockchain_connected": false,
        "coordinator_address": "",
        "bandwidth_bps": 0_u64
    });

    JsonRpcResponse::success(id.clone(), result)
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

    // Map status string to client ProofPhase variant name
    let (generating, phase) = match entry {
        Some(e) => {
            let p = match e.status.to_lowercase().as_str() {
                "committed" | "complete" | "verified" => "Complete",
                "generating" | "proving" => "ProofComputation",
                "witness" | "witness_generation" => "WitnessGeneration",
                "synthesis" => "CircuitSynthesis",
                "failed" | "error" => "Failed",
                _ => "Idle",
            };
            let gen = matches!(e.status.to_lowercase().as_str(), "generating" | "proving" | "witness" | "synthesis");
            (gen, p)
        }
        None => (false, "Idle"),
    };

    let result = serde_json::json!({
        "generating": generating,
        "phase": phase,
        "progress_percent": if phase == "Complete" { 100_u8 } else { 0_u8 },
        "constraints_satisfied": 0_u64,
        "total_constraints": 0_u64,
        "elapsed_ms": 0_u64,
        "estimated_remaining_ms": 0_u64,
        "memory_usage_bytes": 0_u64,
        "gpu_accelerated": false,
        "error_bound": 0.0_f64
    });

    JsonRpcResponse::success(id.clone(), result)
}

fn handle_get_workers(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let model_id = *state.model_id.read();
    let is_training = snap.current_round.is_some();

    let workers: Vec<serde_json::Value> = snap
        .workers
        .iter()
        .map(|w| {
            let status = map_worker_status(&w.status);
            serde_json::json!({
                "id": w.id,
                "address": format!("0x{:040x}", 0_u64),
                "status": status,
                "stake": 0.0_f64,
                "proofs_submitted": w.rounds_completed,
                "proofs_verified": w.rounds_completed,
                "proofs_rejected": 0_u64,
                "reputation": 1.0_f64,
                "is_training": is_training && status == "Training",
                "assigned_model": model_id,
                "last_activity": 0_i64
            })
        })
        .collect();

    JsonRpcResponse::success(id.clone(), serde_json::json!(workers))
}

fn handle_submit_proof(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    // === Rate limiting ===
    // Use model_id as caller identity (or "anonymous" if not provided yet)
    let caller_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .map(|m| format!("model-{}", m))
        .unwrap_or_else(|| "anonymous".to_string());

    {
        let mut limiter = state.rate_limiter.write();
        if !limiter.check(&caller_id) {
            return JsonRpcResponse::error(
                id.clone(),
                -32005,
                "Rate limit exceeded: too many proof submissions. Try again later.".to_string(),
            );
        }
    }

    // === Queue size check ===
    {
        let queue = state.proof_queue.read();
        if queue.len() >= MAX_PROOF_QUEUE_SIZE {
            return JsonRpcResponse::error(
                id.clone(),
                -32006,
                format!("Proof queue full ({} pending). Try again later.", MAX_PROOF_QUEUE_SIZE),
            );
        }
    }

    let model_id = match params.get("model_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "model_id is required (u64)"),
    };
    let round_id = match params.get("round_id").and_then(|v| v.as_u64()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "round_id is required (u64)"),
    };

    // Accept proof bytes as either hex string or array of u8 (client sends Vec<u8>)
    let proof_hex = if let Some(s) = params.get("proof").and_then(|v| v.as_str()) {
        s.to_string()
    } else if let Some(arr) = params.get("proof").and_then(|v| v.as_array()) {
        // Client sends Vec<u8> as JSON array of numbers
        let bytes: Vec<u8> = arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as u8))
            .collect();
        format!("0x{}", hex::encode(&bytes))
    } else {
        return JsonRpcResponse::invalid_params(id.clone(), "proof is required (hex string or byte array)");
    };

    // === Proof size validation ===
    let proof_stripped = proof_hex.strip_prefix("0x").unwrap_or(&proof_hex);
    let proof_byte_len = proof_stripped.len() / 2;
    if proof_byte_len > MAX_PROOF_SIZE {
        return JsonRpcResponse::invalid_params(
            id.clone(),
            &format!("proof too large: {} bytes (max {})", proof_byte_len, MAX_PROOF_SIZE),
        );
    }

    let public_inputs: Vec<String> = match params.get("public_inputs") {
        Some(serde_json::Value::Array(arr)) => {
            // === Public inputs count validation ===
            if arr.len() > MAX_PUBLIC_INPUTS {
                return JsonRpcResponse::invalid_params(
                    id.clone(),
                    &format!("too many public inputs: {} (max {})", arr.len(), MAX_PUBLIC_INPUTS),
                );
            }
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        }
        _ => return JsonRpcResponse::invalid_params(id.clone(), "public_inputs is required (array of hex strings)"),
    };

    // Validate hex encoding
    if hex::decode(proof_stripped).is_err() {
        return JsonRpcResponse::invalid_params(id.clone(), "proof is not valid hex");
    }

    // Compute a hash for tracking
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(proof_stripped.as_bytes());
    let proof_hash = format!("0x{}", hex::encode(&hasher.finalize()[..16]));

    let queued = QueuedProof {
        model_id,
        round_id,
        proof_hex,
        public_inputs_hex: public_inputs,
        submitted: false,
        tx_hash: None,
    };

    state.proof_queue.write().push(queued);

    // Client expects just the proof hash string
    JsonRpcResponse::success(id.clone(), serde_json::json!(proof_hash))
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

    let snap = state.snapshot.read();
    let current_round = snap.current_round.as_ref().map(|r| r.round_id).unwrap_or(0);
    let is_active = snap.current_round.is_some() || snap.completed_rounds > 0;

    let current_commitment = snap
        .current_round
        .as_ref()
        .and_then(|r| r.commitment_hash.clone())
        .unwrap_or_else(|| {
            let agg = state.aggregation_results.read();
            if let Some(latest) = agg.last() {
                format!("0x{}", hex::encode(latest.merkle_root))
            } else {
                String::new()
            }
        });

    // Return client-compatible ModelInfo shape (12 fields)
    let result = serde_json::json!({
        "id": model_id,
        "name": format!("model-{}", model_id),
        "ipfs_hash": "",
        "architecture": "",
        "parameter_count": 0_u64,
        "current_commitment": current_commitment,
        "owner": "",
        "min_stake": 0.0_f64,
        "training_active": is_active,
        "current_round": current_round,
        "accumulated_error": 0.0_f64,
        "created_at": 0_i64
    });

    JsonRpcResponse::success(id.clone(), result)
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
            return JsonRpcResponse::error(
                id.clone(),
                -32000,
                "Training already in progress".to_string(),
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
            JsonRpcResponse::success(id.clone(), serde_json::json!(true))
        }
        Err(_) => {
            JsonRpcResponse::error(
                id.clone(),
                -32000,
                "No training orchestrator listening".to_string(),
            )
        }
    }
}

fn handle_stop_training(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    match state.stop_trigger_tx.send(()) {
        Ok(_) => {
            JsonRpcResponse::success(id.clone(), serde_json::json!(true))
        }
        Err(_) => {
            JsonRpcResponse::error(
                id.clone(),
                -32000,
                "No training session active or no listeners".to_string(),
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

// Old response types removed — handlers now return client-compatible JSON directly

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

    // First check round_weights for real data from completed training
    let weights = state.round_weights.read();
    if let Some(entry) = weights.iter().find(|w| w.round_id == round_id) {
        let result = serde_json::json!({
            "available": true,
            "round_id": entry.round_id,
            "worker_count": entry.num_contributors as u64,
            "loss": entry.loss,
            "error_bound": entry.error_bound,
            "model_dims": null,
            "step_number": entry.steps_completed
        });
        return JsonRpcResponse::success(id.clone(), result);
    }

    // Fall back to latest completed round's data
    if let Some(latest) = weights.last() {
        let result = serde_json::json!({
            "available": true,
            "round_id": latest.round_id,
            "worker_count": latest.num_contributors as u64,
            "loss": latest.loss,
            "error_bound": latest.error_bound,
            "model_dims": null,
            "step_number": latest.steps_completed
        });
        return JsonRpcResponse::success(id.clone(), result);
    }

    // Check aggregation results as fallback
    let agg_results = state.aggregation_results.read();
    let has_agg = agg_results.iter().any(|r| r.round_id == round_id);

    let result = serde_json::json!({
        "available": has_agg || snap.completed_rounds > 0,
        "round_id": round_id,
        "worker_count": snap.available_workers as u64,
        "loss": 0.0_f64,
        "error_bound": 0.0_f64,
        "model_dims": null,
        "step_number": round_id
    });

    JsonRpcResponse::success(id.clone(), result)
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
    let (accepted, message) = match state.round_trigger_tx.send(()) {
        Ok(_) => (true, format!("Proof generation triggered for round {}", round_id)),
        Err(_) => (false, "No training orchestrator listening for proof generation".to_string()),
    };

    // Return client-compatible GenerateProofAck shape
    let result = serde_json::json!({
        "accepted": accepted,
        "message": message,
        "round_id": round_id
    });

    JsonRpcResponse::success(id.clone(), result)
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

    // Client expects just the model_id number
    JsonRpcResponse::success(id.clone(), serde_json::json!(model_id))
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

    // Client expects just a String acknowledgment
    JsonRpcResponse::success(
        id.clone(),
        serde_json::json!(format!(
            "Stake request acknowledged for model {}. On-chain staking must be performed via the coordinator contract.",
            model_id
        )),
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

    // Client expects just a String acknowledgment
    JsonRpcResponse::success(
        id.clone(),
        serde_json::json!(format!(
            "Unstake request acknowledged for model {}. On-chain unstaking must be performed via the coordinator contract.",
            model_id
        )),
    )
}

// ============================================================================
// New handlers for client compatibility
// ============================================================================

fn handle_health(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let uptime = state.start_time.elapsed().as_secs();
    let result = serde_json::json!({
        "healthy": true,
        "components": {},
        "last_check": 0_i64,
        "uptime_secs": uptime,
        "version": "0.1.0"
    });
    JsonRpcResponse::success(id.clone(), result)
}

fn handle_capabilities(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let is_worker = state.node_role == "worker";
    let is_aggregator = state.node_role == "aggregator";
    let result = serde_json::json!({
        "can_train": is_worker || is_aggregator,
        "can_aggregate": is_aggregator,
        "can_prove": true,
        "can_verify": true,
        "has_gpu": false,
        "available_memory": 0_u64,
        "supported_proof_types": ["halo2-kzg"],
        "max_model_params": 1_000_000_u64
    });
    JsonRpcResponse::success(id.clone(), result)
}

fn handle_get_training_progress(_state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    // Node doesn't track per-round progress history; return empty array
    JsonRpcResponse::success(id.clone(), serde_json::json!([]))
}

fn handle_verify_proof(_state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    // Proof verification is on-chain; node returns true to acknowledge
    JsonRpcResponse::success(id.clone(), serde_json::json!(true))
}

fn handle_list_models(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let model_id = *state.model_id.read();
    match model_id {
        Some(mid) => {
            let snap = state.snapshot.read();
            let current_round = snap.current_round.as_ref().map(|r| r.round_id).unwrap_or(0);
            let is_active = snap.current_round.is_some() || snap.completed_rounds > 0;
            let result = serde_json::json!([{
                "id": mid,
                "name": format!("model-{}", mid),
                "ipfs_hash": "",
                "architecture": "",
                "parameter_count": 0_u64,
                "current_commitment": "",
                "owner": "",
                "min_stake": 0.0_f64,
                "training_active": is_active,
                "current_round": current_round,
                "accumulated_error": 0.0_f64,
                "created_at": 0_i64
            }]);
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::success(id.clone(), serde_json::json!([])),
    }
}

fn handle_get_current_round(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let model_id = state.model_id.read().unwrap_or(0);
    match &snap.current_round {
        Some(r) => {
            let result = serde_json::json!({
                "round_id": r.round_id,
                "model_id": model_id,
                "started_at": 0_i64,
                "deadline": 0_i64,
                "completed": false,
                "proofs_submitted": r.gradients_received as u32,
                "proofs_verified": r.gradients_received as u32,
                "participants": [],
                "prev_commitment": "",
                "new_commitment": null,
                "loss": null,
                "error_delta": null
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => {
            // Return a default round with completed=true for the last completed round
            let result = serde_json::json!({
                "round_id": snap.completed_rounds,
                "model_id": model_id,
                "started_at": 0_i64,
                "deadline": 0_i64,
                "completed": true,
                "proofs_submitted": 0_u32,
                "proofs_verified": 0_u32,
                "participants": [],
                "prev_commitment": "",
                "new_commitment": null,
                "loss": null,
                "error_delta": null
            });
            JsonRpcResponse::success(id.clone(), result)
        }
    }
}

fn handle_get_round(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let round_id = params
        .get("round_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .or_else(|| *state.model_id.read())
        .unwrap_or(0);

    let snap = state.snapshot.read();

    // Check if this is the current round
    let is_current = snap.current_round.as_ref().map(|r| r.round_id) == Some(round_id);

    let (proofs_submitted, completed) = if is_current {
        let ps = snap.current_round.as_ref().map(|r| r.gradients_received).unwrap_or(0);
        (ps as u32, false)
    } else {
        // Check aggregation results
        let agg = state.aggregation_results.read();
        let entry = agg.iter().find(|e| e.round_id == round_id);
        match entry {
            Some(e) => (e.proofs_aggregated as u32, true),
            None => (0_u32, round_id <= snap.completed_rounds),
        }
    };

    let result = serde_json::json!({
        "round_id": round_id,
        "model_id": model_id,
        "started_at": 0_i64,
        "deadline": 0_i64,
        "completed": completed,
        "proofs_submitted": proofs_submitted,
        "proofs_verified": proofs_submitted,
        "participants": [],
        "prev_commitment": "",
        "new_commitment": null,
        "loss": null,
        "error_delta": null
    });

    JsonRpcResponse::success(id.clone(), result)
}

fn handle_get_worker(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let worker_id = match params.get("worker_id").and_then(|v| v.as_str()) {
        Some(v) => v,
        None => return JsonRpcResponse::invalid_params(id.clone(), "worker_id is required (string)"),
    };

    let snap = state.snapshot.read();
    let model_id = *state.model_id.read();
    let is_training = snap.current_round.is_some();

    match snap.workers.iter().find(|w| w.id == worker_id) {
        Some(w) => {
            let status = map_worker_status(&w.status);
            let result = serde_json::json!({
                "id": w.id,
                "address": format!("0x{:040x}", 0_u64),
                "status": status,
                "stake": 0.0_f64,
                "proofs_submitted": w.rounds_completed,
                "proofs_verified": w.rounds_completed,
                "proofs_rejected": 0_u64,
                "reputation": 1.0_f64,
                "is_training": is_training && status == "Training",
                "assigned_model": model_id,
                "last_activity": 0_i64
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32000,
            format!("Worker not found: {}", worker_id),
        ),
    }
}

fn handle_get_self_worker(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    let snap = state.snapshot.read();
    let model_id = *state.model_id.read();
    let is_training = snap.current_round.is_some();

    // Return the first worker, or a synthetic entry representing this node
    let result = if let Some(w) = snap.workers.first() {
        let status = map_worker_status(&w.status);
        serde_json::json!({
            "id": w.id,
            "address": format!("0x{:040x}", 0_u64),
            "status": status,
            "stake": 0.0_f64,
            "proofs_submitted": w.rounds_completed,
            "proofs_verified": w.rounds_completed,
            "proofs_rejected": 0_u64,
            "reputation": 1.0_f64,
            "is_training": is_training && status == "Training",
            "assigned_model": model_id,
            "last_activity": 0_i64
        })
    } else {
        serde_json::json!({
            "id": "self",
            "address": format!("0x{:040x}", 0_u64),
            "status": if is_training { "Training" } else { "Idle" },
            "stake": 0.0_f64,
            "proofs_submitted": 0_u64,
            "proofs_verified": 0_u64,
            "proofs_rejected": 0_u64,
            "reputation": 1.0_f64,
            "is_training": is_training,
            "assigned_model": model_id,
            "last_activity": 0_i64
        })
    };

    JsonRpcResponse::success(id.clone(), result)
}

fn handle_get_staking_info(_state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    // Staking is on-chain; return defaults
    let result = serde_json::json!({
        "total_staked": 0.0_f64,
        "your_stake": 0.0_f64,
        "lock_until": 0_i64,
        "pending_rewards": 0.0_f64,
        "total_rewards_claimed": 0.0_f64,
        "is_locked": false,
        "slashing_events": []
    });
    JsonRpcResponse::success(id.clone(), result)
}

fn handle_claim_rewards(_state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    // Reward claiming is on-chain; return acknowledgment
    JsonRpcResponse::success(
        id.clone(),
        serde_json::json!("Claim request acknowledged. On-chain reward claiming must be performed via the Rewards contract."),
    )
}

// ============================================================================
// Result Distribution Handlers
// ============================================================================

fn handle_get_model_weights(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .or_else(|| *state.model_id.read());
    let round_id = params.get("round_id").and_then(|v| v.as_u64());

    let weights = state.round_weights.read();

    let entry = match round_id {
        Some(rid) => weights.iter().find(|w| {
            w.round_id == rid && model_id.map_or(true, |mid| w.model_id == mid)
        }),
        None => {
            // Return the latest round's weights
            weights.iter().filter(|w| model_id.map_or(true, |mid| w.model_id == mid)).last()
        }
    };

    match entry {
        Some(e) => {
            let result = serde_json::json!({
                "available": true,
                "round_id": e.round_id,
                "model_id": e.model_id,
                "commitment": format!("0x{}", hex::encode(e.commitment)),
                "size_bytes": e.weight_bytes.len(),
                "loss": e.loss,
                "error_bound": e.error_bound,
                "steps_completed": e.steps_completed,
                "num_contributors": e.num_contributors,
                "completed_at": e.completed_at,
                "tx_hash": e.tx_hash,
                "weight_bytes": e.weight_bytes,
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => {
            let result = serde_json::json!({
                "available": false,
                "round_id": round_id.unwrap_or(0),
                "model_id": model_id.unwrap_or(0),
                "commitment": null,
                "size_bytes": 0,
                "loss": 0.0,
                "error_bound": 0.0,
                "steps_completed": 0,
                "num_contributors": 0,
                "completed_at": 0,
                "tx_hash": null,
                "weight_bytes": [],
            });
            JsonRpcResponse::success(id.clone(), result)
        }
    }
}

fn handle_get_training_history(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .or_else(|| *state.model_id.read())
        .unwrap_or(0);

    let weights = state.round_weights.read();

    let rounds: Vec<serde_json::Value> = weights
        .iter()
        .filter(|w| w.model_id == model_id)
        .map(|w| {
            serde_json::json!({
                "round_id": w.round_id,
                "commitment": format!("0x{}", hex::encode(w.commitment)),
                "loss": w.loss,
                "error_bound": w.error_bound,
                "steps_completed": w.steps_completed,
                "num_contributors": w.num_contributors,
                "completed_at": w.completed_at,
                "tx_hash": w.tx_hash,
            })
        })
        .collect();

    let total = rounds.len() as u64;
    JsonRpcResponse::success(id.clone(), serde_json::json!({
        "model_id": model_id,
        "rounds": rounds,
        "total_rounds": total,
    }))
}

fn handle_get_training_report(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .or_else(|| *state.model_id.read())
        .unwrap_or(0);

    let weights = state.round_weights.read();
    let model_rounds: Vec<&RoundWeightEntry> = weights
        .iter()
        .filter(|w| w.model_id == model_id)
        .collect();

    if model_rounds.is_empty() {
        return JsonRpcResponse::error(
            id.clone(),
            -32000,
            format!("No training data for model {}", model_id),
        );
    }

    let loss_values: Vec<f64> = model_rounds.iter().map(|r| r.loss).collect();
    let error_values: Vec<f64> = model_rounds.iter().map(|r| r.error_bound).collect();

    let final_loss = loss_values.last().copied().unwrap_or(0.0);
    let final_error = error_values.last().copied().unwrap_or(0.0);
    let final_commitment = model_rounds.last()
        .map(|r| format!("0x{}", hex::encode(r.commitment)))
        .unwrap_or_default();

    let per_round: Vec<serde_json::Value> = model_rounds
        .iter()
        .map(|r| {
            serde_json::json!({
                "round_id": r.round_id,
                "loss": r.loss,
                "error_bound": r.error_bound,
                "steps_completed": r.steps_completed,
                "num_contributors": r.num_contributors,
                "commitment": format!("0x{}", hex::encode(r.commitment)),
                "tx_hash": r.tx_hash,
                "completed_at": r.completed_at,
            })
        })
        .collect();

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let result = serde_json::json!({
        "model_id": model_id,
        "total_rounds": model_rounds.len(),
        "final_loss": final_loss,
        "final_error_bound": final_error,
        "final_commitment": final_commitment,
        "loss_curve": loss_values,
        "error_curve": error_values,
        "per_round": per_round,
        "generated_at": now_secs,
    });

    JsonRpcResponse::success(id.clone(), result)
}

// ============================================================================
// Download Model Handler
// ============================================================================

fn handle_download_model(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let model_id = params
        .get("model_id")
        .and_then(|v| v.as_u64())
        .or_else(|| *state.model_id.read());
    let round_id = params.get("round_id").and_then(|v| v.as_u64());

    let weights = state.round_weights.read();

    let entry = match round_id {
        Some(rid) => weights.iter().find(|w| {
            w.round_id == rid && model_id.map_or(true, |mid| w.model_id == mid)
        }),
        None => {
            weights.iter().filter(|w| model_id.map_or(true, |mid| w.model_id == mid)).last()
        }
    };

    match entry {
        Some(e) => {
            let result = serde_json::json!({
                "available": true,
                "round_id": e.round_id,
                "model_id": e.model_id,
                "commitment": format!("0x{}", hex::encode(e.commitment)),
                "weight_bytes": e.weight_bytes,
                "size_bytes": e.weight_bytes.len(),
                "loss": e.loss,
                "error_bound": e.error_bound,
                "steps_completed": e.steps_completed,
                "num_contributors": e.num_contributors,
                "completed_at": e.completed_at,
                "tx_hash": e.tx_hash,
                "format": "helix-checkpoint-v1",
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32001,
            format!(
                "No model found for model_id={:?}, round_id={:?}",
                model_id, round_id,
            ),
        ),
    }
}

// ============================================================================
// Worker Daemon RPC Handlers
// ============================================================================

fn handle_get_worker_status(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    match &state.worker_daemon {
        Some(daemon) => {
            let status = daemon.status();
            let result = serde_json::json!({
                "auto_join": status.auto_join,
                "in_round": status.in_round,
                "active_round": status.active_round.map(|r| serde_json::json!({
                    "round_id": r.round_id,
                    "model_id": r.model_id,
                    "phase": r.phase,
                    "steps_completed": r.steps_completed,
                    "steps_required": r.steps_required,
                    "elapsed_secs": r.elapsed_secs,
                })),
                "total_rounds": status.total_rounds,
                "rounds_succeeded": status.rounds_succeeded,
                "success_rate": status.success_rate,
                "reputation_score": status.reputation_score,
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32000,
            "Worker daemon not available (node may not be a worker)".to_string(),
        ),
    }
}

fn handle_get_earnings(state: &RpcState, id: &serde_json::Value) -> JsonRpcResponse {
    match &state.worker_daemon {
        Some(daemon) => {
            let earnings = daemon.earnings();
            let result = serde_json::json!({
                "total_earnings_wei": earnings.total_earnings_wei.to_string(),
                "rounds_participated": earnings.rounds_participated,
                "rounds_succeeded": earnings.rounds_succeeded,
                "rounds_failed": earnings.rounds_failed,
                "total_proofs_submitted": earnings.total_proofs_submitted,
                "total_steps_completed": earnings.total_steps_completed,
                "avg_earnings_per_round_wei": earnings.avg_earnings_per_round_wei.to_string(),
                "success_rate": earnings.success_rate,
                "reputation_score": earnings.reputation_score,
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32000,
            "Worker daemon not available (node may not be a worker)".to_string(),
        ),
    }
}

fn handle_set_auto_join(
    state: &RpcState,
    params: &serde_json::Value,
    id: &serde_json::Value,
) -> JsonRpcResponse {
    let enabled = match params.get("enabled").and_then(|v| v.as_bool()) {
        Some(b) => b,
        None => {
            return JsonRpcResponse::error(
                id.clone(),
                -32602,
                "Missing or invalid 'enabled' parameter (expected boolean)".to_string(),
            );
        }
    };

    match &state.worker_daemon {
        Some(daemon) => {
            let previous = daemon.set_auto_join(enabled);
            let result = serde_json::json!({
                "auto_join": enabled,
                "previous": previous,
            });
            JsonRpcResponse::success(id.clone(), result)
        }
        None => JsonRpcResponse::error(
            id.clone(),
            -32000,
            "Worker daemon not available (node may not be a worker)".to_string(),
        ),
    }
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
        rate_limiter: Arc::new(RwLock::new(RpcRateLimiter::default())),
        round_weights: Arc::new(RwLock::new(Vec::new())),
        worker_daemon: None,
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
            rpc_addr: "127.0.0.1:9002".to_string(),
            rate_limiter: Arc::new(RwLock::new(RpcRateLimiter::default())),
            round_weights: Arc::new(RwLock::new(Vec::new())),
            worker_daemon: None,
        })
    }

    fn null_id() -> serde_json::Value {
        serde_json::Value::Number(1.into())
    }

    // ---- Client-compatible shape tests ----

    #[test]
    fn test_get_training_status_client_shape() {
        let state = create_test_state();
        let response = handle_get_training_status(&state, &null_id());
        let v = response.result.unwrap();

        // Must have all 11 fields that client TrainingStatus expects
        assert_eq!(v["active"], false);
        assert_eq!(v["phase"], "Idle");
        assert!(v["current_round"].is_number());
        assert!(v["total_rounds"].is_number());
        assert!(v["current_loss"].is_number());
        assert!(v["accumulated_error"].is_number());
        assert!(v["max_error_bound"].is_number());
        assert!(v["round_elapsed_ms"].is_number());
        assert!(v["estimated_remaining_ms"].is_number());
        assert_eq!(v["model_id"], 1);
        assert!(v["started_at"].is_number());
    }

    #[test]
    fn test_get_network_status_client_shape() {
        let state = create_test_state();
        let response = handle_get_network_status(&state, &null_id());
        let v = response.result.unwrap();

        assert_eq!(v["peer_count"], 3);
        assert_eq!(v["active_workers"], 2);
        assert_eq!(v["active_aggregators"], 1); // aggregator role
        assert!(v["avg_latency_ms"].is_number());
        assert!(v["block_height"].is_number());
        assert!(v["chain_id"].is_number());
        assert_eq!(v["blockchain_connected"], false);
        assert!(v["coordinator_address"].is_string());
        assert!(v["bandwidth_bps"].is_number());
    }

    #[test]
    fn test_get_proof_status_client_shape() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 1});
        let response = handle_get_proof_status(&state, &params, &null_id());
        let v = response.result.unwrap();

        // committed → Complete
        assert_eq!(v["generating"], false);
        assert_eq!(v["phase"], "Complete");
        assert_eq!(v["progress_percent"], 100);
        assert!(v["constraints_satisfied"].is_number());
        assert!(v["total_constraints"].is_number());
        assert!(v["elapsed_ms"].is_number());
        assert!(v["estimated_remaining_ms"].is_number());
        assert!(v["memory_usage_bytes"].is_number());
        assert_eq!(v["gpu_accelerated"], false);
        assert!(v["error_bound"].is_number());
    }

    #[test]
    fn test_get_proof_status_not_found_client_shape() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 999});
        let response = handle_get_proof_status(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["phase"], "Idle");
        assert_eq!(v["generating"], false);
    }

    #[test]
    fn test_get_workers_client_shape() {
        let state = create_test_state();
        let response = handle_get_workers(&state, &null_id());
        let v = response.result.unwrap();
        // Default snapshot has no workers
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 0);
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
    fn test_submit_proof_returns_hash_string() {
        let state = create_test_state();
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "0xdeadbeef",
            "public_inputs": ["0x01", "0x02", "0x03", "0x04", "0x05", "0x06", "0x07"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        let v = response.result.unwrap();

        // Client expects a plain string (the proof hash)
        assert!(v.is_string());
        assert!(v.as_str().unwrap().starts_with("0x"));

        // Verify proof was queued
        let queue = state.proof_queue.read();
        assert_eq!(queue.len(), 1);
        assert_eq!(queue[0].model_id, 1);
        assert_eq!(queue[0].round_id, 1);
    }

    #[test]
    fn test_submit_proof_accepts_byte_array() {
        let state = create_test_state();
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 2,
            "proof": [0xde, 0xad, 0xbe, 0xef],
            "public_inputs": ["0x01"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.result.is_some());
        let v = response.result.unwrap();
        assert!(v.is_string());
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
    fn test_get_model_state_client_shape() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_get_model_state(&state, &params, &null_id());
        let v = response.result.unwrap();

        // Must have all 12 fields that client ModelInfo expects
        assert_eq!(v["id"], 1);
        assert!(v["name"].is_string());
        assert!(v["ipfs_hash"].is_string());
        assert!(v["architecture"].is_string());
        assert!(v["parameter_count"].is_number());
        assert!(v["current_commitment"].is_string());
        assert!(v["owner"].is_string());
        assert!(v["min_stake"].is_number());
        assert!(v["training_active"].is_boolean());
        assert!(v["current_round"].is_number());
        assert!(v["accumulated_error"].is_number());
        assert!(v["created_at"].is_number());
    }

    #[test]
    fn test_get_model_state_uses_active_model() {
        let state = create_test_state();
        let params = serde_json::json!({});
        let response = handle_get_model_state(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["id"], 1);
    }

    #[test]
    fn test_start_training_returns_success() {
        let state = create_test_state();
        let mut _rx = state.round_trigger_tx.subscribe();
        let params = serde_json::json!({"model_id": 2});
        let response = handle_start_training(&state, &params, &null_id());
        assert!(response.result.is_some());
        assert_eq!(response.result.unwrap(), true);

        // Verify model_id was updated
        assert_eq!(*state.model_id.read(), Some(2));
    }

    #[test]
    fn test_stop_training_returns_success() {
        let state = create_test_state();
        let mut _rx = state.stop_trigger_tx.subscribe();
        let response = handle_stop_training(&state, &null_id());
        assert!(response.result.is_some());
        assert_eq!(response.result.unwrap(), true);
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
        assert_eq!(result.rpc_endpoint, "127.0.0.1:9002");
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
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_create_default_rpc_state() {
        let state = create_default_rpc_state("worker", "0.0.0.0:9002");
        assert_eq!(state.node_role, "worker");
        assert_eq!(state.rpc_addr, "0.0.0.0:9002");
        assert!(state.model_id.read().is_none());
    }

    // ---- New handler tests ----

    #[test]
    fn test_health_client_shape() {
        let state = create_test_state();
        let response = handle_health(&state, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["healthy"], true);
        assert!(v["components"].is_object());
        assert!(v["last_check"].is_number());
        assert!(v["uptime_secs"].is_number());
        assert_eq!(v["version"], "0.1.0");
    }

    #[test]
    fn test_capabilities_client_shape() {
        let state = create_test_state();
        let response = handle_capabilities(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v["can_train"].is_boolean());
        assert_eq!(v["can_aggregate"], true); // aggregator role
        assert!(v["can_prove"].is_boolean());
        assert!(v["can_verify"].is_boolean());
        assert_eq!(v["has_gpu"], false);
        assert!(v["available_memory"].is_number());
        assert!(v["supported_proof_types"].is_array());
        assert!(v["max_model_params"].is_number());
    }

    #[test]
    fn test_get_training_progress_returns_empty() {
        let state = create_test_state();
        let response = handle_get_training_progress(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_verify_proof_returns_bool() {
        let state = create_test_state();
        let response = handle_verify_proof(&state, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v, true);
    }

    #[test]
    fn test_list_models_with_active_model() {
        let state = create_test_state();
        let response = handle_list_models(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_array());
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], 1);
        assert!(arr[0]["name"].is_string());
    }

    #[test]
    fn test_list_models_without_active_model() {
        let state = create_test_state();
        *state.model_id.write() = None;
        let response = handle_list_models(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_get_current_round_client_shape() {
        let state = create_test_state();
        let response = handle_get_current_round(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v["round_id"].is_number());
        assert!(v["model_id"].is_number());
        assert!(v["started_at"].is_number());
        assert!(v["deadline"].is_number());
        assert!(v["completed"].is_boolean());
        assert!(v["proofs_submitted"].is_number());
        assert!(v["proofs_verified"].is_number());
        assert!(v["participants"].is_array());
    }

    #[test]
    fn test_get_round_client_shape() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 1, "model_id": 1});
        let response = handle_get_round(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["round_id"], 1);
        assert_eq!(v["model_id"], 1);
        assert_eq!(v["completed"], true); // round 1 < completed_rounds 5
    }

    #[test]
    fn test_get_worker_found() {
        let state = create_test_state();
        // Add a worker to the snapshot
        {
            let mut snap = state.snapshot.write();
            snap.workers.push(crate::api::http::WorkerInfo {
                id: "worker-1".to_string(),
                status: "idle".to_string(),
                rounds_completed: 10,
            });
        }
        let params = serde_json::json!({"worker_id": "worker-1"});
        let response = handle_get_worker(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["id"], "worker-1");
        assert_eq!(v["status"], "Idle");
        assert_eq!(v["proofs_submitted"], 10);
    }

    #[test]
    fn test_get_worker_not_found() {
        let state = create_test_state();
        let params = serde_json::json!({"worker_id": "nonexistent"});
        let response = handle_get_worker(&state, &params, &null_id());
        assert!(response.error.is_some());
    }

    #[test]
    fn test_get_self_worker_client_shape() {
        let state = create_test_state();
        let response = handle_get_self_worker(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v["id"].is_string());
        assert!(v["address"].is_string());
        assert!(v["status"].is_string());
        assert!(v["stake"].is_number());
        assert!(v["reputation"].is_number());
    }

    #[test]
    fn test_get_staking_info_client_shape() {
        let state = create_test_state();
        let response = handle_get_staking_info(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v["total_staked"].is_number());
        assert!(v["your_stake"].is_number());
        assert!(v["lock_until"].is_number());
        assert!(v["pending_rewards"].is_number());
        assert!(v["total_rewards_claimed"].is_number());
        assert!(v["is_locked"].is_boolean());
        assert!(v["slashing_events"].is_array());
    }

    #[test]
    fn test_claim_rewards_returns_string() {
        let state = create_test_state();
        let response = handle_claim_rewards(&state, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_string());
    }

    #[test]
    fn test_register_model_returns_u64() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 42});
        let response = handle_register_model(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v, 42);
        assert_eq!(*state.model_id.read(), Some(42));
    }

    #[test]
    fn test_generate_proof_client_shape() {
        let state = create_test_state();
        let mut _rx = state.round_trigger_tx.subscribe();
        let params = serde_json::json!({"round_id": 5, "model_id": 1});
        let response = handle_generate_proof(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["accepted"], true);
        assert!(v["message"].is_string());
        assert_eq!(v["round_id"], 5);
    }

    #[test]
    fn test_stake_returns_string() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1, "amount_eth": 1.0});
        let response = handle_stake(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_string());
    }

    #[test]
    fn test_unstake_returns_string() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_unstake(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert!(v.is_string());
    }

    // ---- Result distribution tests ----

    fn state_with_weights() -> Arc<RpcState> {
        let state = create_test_state();
        {
            let mut weights = state.round_weights.write();
            weights.push(RoundWeightEntry {
                round_id: 0,
                model_id: 1,
                commitment: [0xAA; 32],
                weight_bytes: vec![1, 2, 3, 4, 5],
                loss: 0.5,
                error_bound: 0.02,
                steps_completed: 100,
                num_contributors: 3,
                completed_at: 1700000000,
                tx_hash: Some("0xabc".to_string()),
            });
            weights.push(RoundWeightEntry {
                round_id: 1,
                model_id: 1,
                commitment: [0xBB; 32],
                weight_bytes: vec![6, 7, 8, 9, 10],
                loss: 0.3,
                error_bound: 0.015,
                steps_completed: 100,
                num_contributors: 3,
                completed_at: 1700001000,
                tx_hash: Some("0xdef".to_string()),
            });
        }
        state
    }

    #[test]
    fn test_get_model_weights_latest() {
        let state = state_with_weights();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_get_model_weights(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["available"], true);
        assert_eq!(v["round_id"], 1);
        assert_eq!(v["size_bytes"], 5);
        assert_eq!(v["loss"], 0.3);
        assert_eq!(v["weight_bytes"], serde_json::json!([6, 7, 8, 9, 10]));
    }

    #[test]
    fn test_get_model_weights_specific_round() {
        let state = state_with_weights();
        let params = serde_json::json!({"model_id": 1, "round_id": 0});
        let response = handle_get_model_weights(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["available"], true);
        assert_eq!(v["round_id"], 0);
        assert_eq!(v["loss"], 0.5);
    }

    #[test]
    fn test_get_model_weights_not_found() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 1, "round_id": 99});
        let response = handle_get_model_weights(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["available"], false);
    }

    #[test]
    fn test_get_training_history() {
        let state = state_with_weights();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_get_training_history(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["model_id"], 1);
        assert_eq!(v["total_rounds"], 2);
        let arr = v["rounds"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["round_id"], 0);
        assert_eq!(arr[0]["loss"], 0.5);
        assert_eq!(arr[1]["round_id"], 1);
        assert_eq!(arr[1]["loss"], 0.3);
    }

    #[test]
    fn test_get_training_history_empty() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 99});
        let response = handle_get_training_history(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["total_rounds"], 0);
        let arr = v["rounds"].as_array().unwrap();
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_get_training_report() {
        let state = state_with_weights();
        let params = serde_json::json!({"model_id": 1});
        let response = handle_get_training_report(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert_eq!(v["model_id"], 1);
        assert_eq!(v["total_rounds"], 2);
        assert_eq!(v["final_loss"], 0.3);
        assert!(v["loss_curve"].is_array());
        assert_eq!(v["loss_curve"].as_array().unwrap().len(), 2);
        assert!(v["per_round"].is_array());
        assert_eq!(v["per_round"].as_array().unwrap().len(), 2);
        assert!(v["generated_at"].is_number());
    }

    #[test]
    fn test_get_training_report_no_data() {
        let state = create_test_state();
        let params = serde_json::json!({"model_id": 99});
        let response = handle_get_training_report(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32000);
    }

    #[test]
    fn test_get_training_result_client_shape() {
        let state = create_test_state();
        let params = serde_json::json!({"round_id": 1});
        let response = handle_get_training_result(&state, &params, &null_id());
        let v = response.result.unwrap();
        assert!(v["available"].is_boolean());
        assert!(v["round_id"].is_number());
        assert!(v["worker_count"].is_number());
        assert!(v["loss"].is_number());
        assert!(v["error_bound"].is_number());
        assert!(v["model_dims"].is_null());
        assert!(v["step_number"].is_number());
    }

    #[test]
    fn test_network_status_alias() {
        // helix_networkStatus should route to the same handler as helix_getNetworkStatus
        let state = create_test_state();
        let r1 = handle_get_network_status(&state, &null_id());
        let v1 = r1.result.unwrap();
        assert_eq!(v1["peer_count"], 3);
        assert_eq!(v1["active_workers"], 2);
    }

    // ---- Rate limiting and DoS protection tests ----

    #[test]
    fn test_rpc_rate_limiter_allows_within_burst() {
        let mut limiter = RpcRateLimiter::new(10.0, 5.0);
        // First 5 should succeed (burst)
        for _ in 0..5 {
            assert!(limiter.check("caller-1"));
        }
        // 6th should fail
        assert!(!limiter.check("caller-1"));
    }

    #[test]
    fn test_rpc_rate_limiter_independent_callers() {
        let mut limiter = RpcRateLimiter::new(10.0, 2.0);
        assert!(limiter.check("caller-1"));
        assert!(limiter.check("caller-1"));
        assert!(!limiter.check("caller-1")); // exhausted
        // Different caller should still work
        assert!(limiter.check("caller-2"));
    }

    #[test]
    fn test_submit_proof_rate_limited() {
        // Create state with very restrictive rate limiter
        let state = create_test_state();
        *state.rate_limiter.write() = RpcRateLimiter::new(0.001, 1.0);

        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "0xdeadbeef",
            "public_inputs": ["0x01"]
        });

        // First request succeeds
        let r1 = handle_submit_proof(&state, &params, &null_id());
        assert!(r1.result.is_some(), "first submit should succeed");

        // Second request should be rate limited
        let r2 = handle_submit_proof(&state, &params, &null_id());
        assert!(r2.error.is_some(), "second submit should be rate limited");
        assert_eq!(r2.error.unwrap().code, -32005);
    }

    #[test]
    fn test_submit_proof_queue_full_rejected() {
        let state = create_test_state();
        // Fill the queue to capacity
        {
            let mut queue = state.proof_queue.write();
            for i in 0..MAX_PROOF_QUEUE_SIZE {
                queue.push(QueuedProof {
                    model_id: 1,
                    round_id: i as u64,
                    proof_hex: "0xab".to_string(),
                    public_inputs_hex: vec![],
                    submitted: false,
                    tx_hash: None,
                });
            }
        }

        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "0xdeadbeef",
            "public_inputs": ["0x01"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32006);
    }

    #[test]
    fn test_submit_proof_oversized_rejected() {
        let state = create_test_state();
        // Create a proof that exceeds MAX_PROOF_SIZE (10MB)
        let huge_proof = format!("0x{}", "ab".repeat(MAX_PROOF_SIZE + 1));
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": huge_proof,
            "public_inputs": ["0x01"]
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602); // invalid params
    }

    #[test]
    fn test_submit_proof_too_many_public_inputs() {
        let state = create_test_state();
        let many_inputs: Vec<String> = (0..MAX_PUBLIC_INPUTS + 1)
            .map(|i| format!("0x{:02x}", i % 256))
            .collect();
        let params = serde_json::json!({
            "model_id": 1,
            "round_id": 1,
            "proof": "0xdeadbeef",
            "public_inputs": many_inputs
        });
        let response = handle_submit_proof(&state, &params, &null_id());
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32602);
    }
}
