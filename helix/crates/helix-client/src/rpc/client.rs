//! HELIX Node RPC Client
//!
//! JSON-RPC client for communicating with helix-node instances.
//! Provides typed methods for all training, proof, and staking operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

/// RPC Error types
#[derive(Debug, Error)]
pub enum RpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Request timeout")]
    Timeout,
    #[error("Node not available: {0}")]
    NodeUnavailable(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("RPC error: {code} - {message}")]
    RpcError { code: i64, message: String },
    #[error("Training not active")]
    TrainingNotActive,
    #[error("Proof generation failed: {0}")]
    ProofFailed(String),
    #[error("Insufficient stake: required {required}, available {available}")]
    InsufficientStake { required: f64, available: f64 },
    #[error("Model not found: {0}")]
    ModelNotFound(u64),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<reqwest::Error> for RpcError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            RpcError::Timeout
        } else if e.is_connect() {
            RpcError::ConnectionFailed(e.to_string())
        } else {
            RpcError::InvalidResponse(e.to_string())
        }
    }
}

/// RPC client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelixRpcConfig {
    /// Node RPC endpoint URL
    pub endpoint: String,
    /// Request timeout
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// Base retry delay in milliseconds (used for exponential backoff)
    pub retry_delay_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Enable request compression
    pub compress: bool,
    /// Authentication token (if required)
    pub auth_token: Option<String>,
}

impl Default for HelixRpcConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:9002/rpc".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            retry_delay_ms: 500,
            max_backoff_ms: 10_000,
            compress: false,
            auth_token: None,
        }
    }
}

impl HelixRpcConfig {
    /// Create config for local development
    pub fn local() -> Self {
        Self::default()
    }

    /// Create config for specific endpoint
    pub fn with_endpoint(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            ..Default::default()
        }
    }
}

/// Training status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    /// Whether training is currently active
    pub active: bool,
    /// Current training phase
    pub phase: TrainingPhase,
    /// Current round number
    pub current_round: u64,
    /// Total rounds configured
    pub total_rounds: u64,
    /// Current loss value
    pub current_loss: f64,
    /// Accumulated error bound
    pub accumulated_error: f64,
    /// Maximum allowed error bound
    pub max_error_bound: f64,
    /// Time elapsed in current round (ms)
    pub round_elapsed_ms: u64,
    /// Estimated time remaining (ms)
    pub estimated_remaining_ms: u64,
    /// Model being trained
    pub model_id: u64,
    /// Training started timestamp
    pub started_at: i64,
}

/// Training phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPhase {
    Initializing,
    LoadingData,
    Forward,
    Backward,
    GradientCompute,
    ProofGeneration,
    ProofSubmission,
    ProofVerification,
    WeightUpdate,
    Checkpointing,
    RoundComplete,
    Idle,
    Failed,
    #[serde(other)]
    Unknown,
}

impl TrainingPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Initializing => "Initializing",
            Self::LoadingData => "Loading Data",
            Self::Forward => "Forward Pass",
            Self::Backward => "Backward Pass",
            Self::GradientCompute => "Computing Gradients",
            Self::ProofGeneration => "Generating Proof",
            Self::ProofSubmission => "Submitting Proof",
            Self::ProofVerification => "Verifying Proof",
            Self::WeightUpdate => "Updating Weights",
            Self::Checkpointing => "Checkpointing",
            Self::RoundComplete => "Round Complete",
            Self::Idle => "Idle",
            Self::Failed => "Failed",
            Self::Unknown => "Unknown",
        }
    }
}

/// Proof generation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatus {
    /// Whether proof generation is in progress
    pub generating: bool,
    /// Current proof generation phase
    pub phase: ProofPhase,
    /// Progress percentage (0-100)
    pub progress_percent: u8,
    /// Number of constraints satisfied
    pub constraints_satisfied: u64,
    /// Total constraints to satisfy
    pub total_constraints: u64,
    /// Time elapsed in proof generation (ms)
    pub elapsed_ms: u64,
    /// Estimated time remaining (ms)
    pub estimated_remaining_ms: u64,
    /// Memory usage (bytes)
    pub memory_usage_bytes: u64,
    /// Whether using GPU acceleration
    pub gpu_accelerated: bool,
    /// Error bound for this proof
    pub error_bound: f64,
}

/// Proof generation phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofPhase {
    Idle,
    WitnessGeneration,
    CircuitSynthesis,
    ProvingKeyLoad,
    CommitmentGeneration,
    ConstraintSatisfaction,
    ProofComputation,
    Serialization,
    Complete,
    Failed,
    #[serde(other)]
    Unknown,
}

impl ProofPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::WitnessGeneration => "Generating Witness",
            Self::CircuitSynthesis => "Synthesizing Circuit",
            Self::ProvingKeyLoad => "Loading Proving Key",
            Self::CommitmentGeneration => "Generating Commitments",
            Self::ConstraintSatisfaction => "Satisfying Constraints",
            Self::ProofComputation => "Computing Proof",
            Self::Serialization => "Serializing",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
            Self::Unknown => "Unknown",
        }
    }
}

/// Network status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    /// Number of connected peers
    pub peer_count: u32,
    /// Number of active workers
    pub active_workers: u32,
    /// Number of active aggregators
    pub active_aggregators: u32,
    /// Network latency (ms)
    pub avg_latency_ms: u64,
    /// Block height on connected chain
    pub block_height: u64,
    /// Chain ID
    pub chain_id: u64,
    /// Whether connected to blockchain RPC
    pub blockchain_connected: bool,
    /// Coordinator contract address
    pub coordinator_address: String,
    /// Network bandwidth usage (bytes/sec)
    pub bandwidth_bps: u64,
}

/// Worker node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub id: String,
    /// Ethereum address
    pub address: String,
    /// Current status
    pub status: WorkerStatus,
    /// Staked amount
    pub stake: f64,
    /// Proofs submitted count
    pub proofs_submitted: u64,
    /// Proofs verified count
    pub proofs_verified: u64,
    /// Proofs rejected count
    pub proofs_rejected: u64,
    /// Reputation score (0.0 - 1.0)
    pub reputation: f64,
    /// Whether currently training
    pub is_training: bool,
    /// Current model assignment
    pub assigned_model: Option<u64>,
    /// Last activity timestamp
    pub last_activity: i64,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    Offline,
    Connecting,
    Syncing,
    Idle,
    Training,
    Proving,
    Waiting,
    Faulted,
    Slashed,
    #[serde(other)]
    Unknown,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: u64,
    /// Model name
    pub name: String,
    /// IPFS hash of model weights
    pub ipfs_hash: String,
    /// Model architecture description
    pub architecture: String,
    /// Parameter count
    pub parameter_count: u64,
    /// Current commitment hash
    pub current_commitment: String,
    /// Owner address
    pub owner: String,
    /// Minimum stake required
    pub min_stake: f64,
    /// Whether training is active
    pub training_active: bool,
    /// Current training round
    pub current_round: u64,
    /// Accumulated error bound
    pub accumulated_error: f64,
    /// Created timestamp
    pub created_at: i64,
}

/// Training round information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundInfo {
    /// Round ID
    pub round_id: u64,
    /// Model ID
    pub model_id: u64,
    /// Round start timestamp
    pub started_at: i64,
    /// Round deadline timestamp
    pub deadline: i64,
    /// Whether round is complete
    pub completed: bool,
    /// Number of proofs submitted
    pub proofs_submitted: u32,
    /// Number of proofs verified
    pub proofs_verified: u32,
    /// Participants in this round
    pub participants: Vec<String>,
    /// Previous model commitment
    pub prev_commitment: String,
    /// New model commitment (after completion)
    pub new_commitment: Option<String>,
    /// Loss achieved this round
    pub loss: Option<f64>,
    /// Error bound delta
    pub error_delta: Option<f64>,
}

/// Training result data from a completed round (used for proof generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResultData {
    /// Whether a completed round result is available.
    pub available: bool,
    /// The round that completed.
    pub round_id: u64,
    /// Number of contributing workers.
    pub worker_count: u64,
    /// Computed loss value.
    pub loss: f64,
    /// Accumulated error bound.
    pub error_bound: f64,
    /// Model dimensions (d_in, d_hid, d_out), if known.
    pub model_dims: Option<(usize, usize, usize)>,
    /// Training step number.
    pub step_number: u64,
}

/// Staking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingInfo {
    /// Total staked amount (ETH)
    pub total_staked: f64,
    /// Your staked amount (ETH)
    pub your_stake: f64,
    /// Lock period end timestamp
    pub lock_until: i64,
    /// Accumulated rewards (ETH)
    pub pending_rewards: f64,
    /// Total rewards claimed (ETH)
    pub total_rewards_claimed: f64,
    /// Whether stake is locked
    pub is_locked: bool,
    /// Slashing history
    pub slashing_events: Vec<SlashingEvent>,
}

/// Slashing event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    /// Event timestamp
    pub timestamp: i64,
    /// Amount slashed (ETH)
    pub amount: f64,
    /// Reason for slashing
    pub reason: String,
    /// Transaction hash
    pub tx_hash: String,
}

/// Proof submission data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSubmission {
    /// Serialized proof data
    pub proof: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<String>,
    /// Model ID
    pub model_id: u64,
    /// Round ID
    pub round_id: u64,
    /// Old state hash (lo, hi)
    pub old_state_hash: (String, String),
    /// New state hash (lo, hi)
    pub new_state_hash: (String, String),
    /// Loss value
    pub loss: String,
    /// Error bound
    pub error_bound: String,
    /// Step number
    pub step_number: u64,
}

impl ProofSubmission {
    /// Creates a `ProofSubmission` from the canonical `EvmProofBundle`.
    ///
    /// Bridges from the prover's on-chain-ready proof format to the RPC
    /// submission format used by `helix_submitProof`.
    pub fn from_evm_bundle(
        bundle: &helix_prover::EvmProofBundle,
        model_id: u64,
        round_id: u64,
    ) -> Self {
        let public_inputs: Vec<String> = bundle
            .evm_public_inputs
            .iter()
            .map(|pi| format!("0x{}", pi.iter().map(|b| format!("{:02x}", b)).collect::<String>()))
            .collect();

        // Extract typed fields from the 8 public inputs (big-endian 32-byte arrays).
        // PI layout: [old_hash_lo, old_hash_hi, new_hash_lo, new_hash_hi, loss, error_bound, step_number, error_checksum]
        let pi = &bundle.evm_public_inputs;

        let old_state_hash = (
            public_inputs.get(0).cloned().unwrap_or_default(),
            public_inputs.get(1).cloned().unwrap_or_default(),
        );
        let new_state_hash = (
            public_inputs.get(2).cloned().unwrap_or_default(),
            public_inputs.get(3).cloned().unwrap_or_default(),
        );
        let loss = public_inputs.get(4).cloned().unwrap_or_default();
        let error_bound = public_inputs.get(5).cloned().unwrap_or_default();

        // Step number from PI[6]: read as big-endian u64 from last 8 bytes.
        let step_number = if pi.len() > 6 {
            let bytes = &pi[6];
            u64::from_be_bytes(bytes[24..32].try_into().unwrap_or([0u8; 8]))
        } else {
            0
        };

        Self {
            proof: bundle.evm_proof.clone(),
            public_inputs,
            model_id,
            round_id,
            old_state_hash,
            new_state_hash,
            loss,
            error_bound,
            step_number,
        }
    }
}

/// Acknowledgment returned when proof generation is triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateProofAck {
    /// Whether the proof generation request was accepted
    pub accepted: bool,
    /// Status message
    pub message: String,
    /// Round ID for the proof
    pub round_id: u64,
}

/// Training progress snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    /// Round number
    pub round: u64,
    /// Loss value
    pub loss: f64,
    /// Error bound
    pub error_bound: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Proof generation time (ms)
    pub proof_time_ms: u64,
    /// Verification time (ms)
    pub verify_time_ms: u64,
    /// Learning rate used
    pub learning_rate: f64,
    /// Gradient norm
    pub gradient_norm: f64,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Can perform training computation
    pub can_train: bool,
    /// Can aggregate gradients
    pub can_aggregate: bool,
    /// Can generate proofs
    pub can_prove: bool,
    /// Can verify proofs
    pub can_verify: bool,
    /// Has GPU acceleration
    pub has_gpu: bool,
    /// Available memory (bytes)
    pub available_memory: u64,
    /// Supported proof types
    pub supported_proof_types: Vec<String>,
    /// Maximum model size (parameters)
    pub max_model_params: u64,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub healthy: bool,
    /// Component statuses
    pub components: HashMap<String, ComponentHealth>,
    /// Last check timestamp
    pub last_check: i64,
    /// Uptime in seconds
    pub uptime_secs: u64,
    /// Version information
    pub version: String,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Whether healthy
    pub healthy: bool,
    /// Status message
    pub message: String,
    /// Response time (ms)
    pub response_time_ms: Option<u64>,
}

/// JSON-RPC request
#[derive(Debug, Serialize)]
struct JsonRpcRequest<T: Serialize> {
    jsonrpc: &'static str,
    method: String,
    params: T,
    id: u64,
}

/// JSON-RPC response
#[derive(Debug, Deserialize)]
struct JsonRpcResponse<T> {
    #[allow(dead_code)]
    jsonrpc: String,
    result: Option<T>,
    error: Option<JsonRpcError>,
    #[allow(dead_code)]
    id: u64,
}

/// JSON-RPC error
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

/// Circuit breaker state for RPC requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcCircuitBreakerState {
    /// Normal operation — requests flow through
    Closed,
    /// Breaker tripped — requests are rejected immediately
    Open,
    /// Testing recovery — one probe request allowed
    HalfOpen,
}

/// Circuit breaker that prevents hammering a down endpoint
#[derive(Debug)]
pub struct RpcCircuitBreaker {
    /// Current state
    pub state: RpcCircuitBreakerState,
    /// Consecutive failure count
    pub failure_count: u32,
    /// Failure threshold to trip breaker
    pub threshold: u32,
    /// Time of last failure
    pub last_failure_time: Option<Instant>,
    /// How long to wait before transitioning Open → HalfOpen
    pub reset_timeout: Duration,
}

impl RpcCircuitBreaker {
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: RpcCircuitBreakerState::Closed,
            failure_count: 0,
            threshold,
            last_failure_time: None,
            reset_timeout,
        }
    }

    /// Check whether a request should be allowed
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            RpcCircuitBreakerState::Closed => true,
            RpcCircuitBreakerState::Open => {
                // Check if reset_timeout has elapsed → transition to HalfOpen
                if let Some(last) = self.last_failure_time {
                    if last.elapsed() >= self.reset_timeout {
                        self.state = RpcCircuitBreakerState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            RpcCircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful request
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.state = RpcCircuitBreakerState::Closed;
    }

    /// Record a failed request (after all retries exhausted)
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());
        if self.failure_count >= self.threshold {
            self.state = RpcCircuitBreakerState::Open;
        }
    }
}

impl Default for RpcCircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

/// HELIX Node RPC Client
pub struct HelixRpcClient {
    /// HTTP client
    client: Client,
    /// Configuration
    config: HelixRpcConfig,
    /// Request counter for JSON-RPC IDs
    request_id: Arc<RwLock<u64>>,
    /// Connection state
    connected: Arc<RwLock<bool>>,
    /// Last successful request timestamp
    last_success: Arc<RwLock<Option<Instant>>>,
    /// Cached node capabilities
    capabilities: Arc<RwLock<Option<NodeCapabilities>>>,
    /// Circuit breaker to avoid hammering downed endpoints
    circuit_breaker: Arc<RwLock<RpcCircuitBreaker>>,
}

impl HelixRpcClient {
    /// Create a new RPC client
    pub fn new(config: HelixRpcConfig) -> Result<Self, RpcError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(4)
            .build()
            .map_err(|e| RpcError::ConnectionFailed(e.to_string()))?;

        Ok(Self {
            client,
            config,
            request_id: Arc::new(RwLock::new(0)),
            connected: Arc::new(RwLock::new(false)),
            last_success: Arc::new(RwLock::new(None)),
            capabilities: Arc::new(RwLock::new(None)),
            circuit_breaker: Arc::new(RwLock::new(RpcCircuitBreaker::default())),
        })
    }

    /// Create client with default config
    pub fn default_client() -> Result<Self, RpcError> {
        Self::new(HelixRpcConfig::default())
    }

    /// Create client for local node
    pub fn local() -> Result<Self, RpcError> {
        Self::new(HelixRpcConfig::local())
    }

    /// Connect to the node and verify availability
    pub async fn connect(&self) -> Result<(), RpcError> {
        let health = self.health_check().await?;

        if !health.healthy {
            return Err(RpcError::NodeUnavailable("Node reports unhealthy".to_string()));
        }

        *self.connected.write().await = true;
        *self.last_success.write().await = Some(Instant::now());

        // Cache capabilities
        if let Ok(caps) = self.get_capabilities().await {
            *self.capabilities.write().await = Some(caps);
        }

        Ok(())
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }

    /// Get a reference to the circuit breaker (for testing / monitoring)
    pub fn circuit_breaker(&self) -> &Arc<RwLock<RpcCircuitBreaker>> {
        &self.circuit_breaker
    }

    /// Send JSON-RPC request with exponential backoff and circuit breaker
    async fn send_request<P: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: P,
    ) -> Result<R, RpcError> {
        // Check circuit breaker before attempting
        {
            let mut cb = self.circuit_breaker.write().await;
            if !cb.allow_request() {
                return Err(RpcError::NodeUnavailable(
                    "Circuit breaker is open — endpoint unavailable".to_string(),
                ));
            }
        }

        let id = {
            let mut counter = self.request_id.write().await;
            *counter += 1;
            *counter
        };

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id,
        };

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                // Exponential backoff with jitter
                let base_delay = self.config.retry_delay_ms * (1u64 << attempt.min(6));
                let capped_delay = base_delay.min(self.config.max_backoff_ms);
                let jitter = rand::random::<u64>() % (capped_delay / 4 + 1);
                let delay = capped_delay + jitter;
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }

            let mut request_builder = self.client
                .post(&self.config.endpoint)
                .header("Content-Type", "application/json");

            if let Some(ref token) = self.config.auth_token {
                request_builder = request_builder.header("Authorization", format!("Bearer {}", token));
            }

            match request_builder.json(&request).send().await {
                Ok(response) => {
                    match response.json::<JsonRpcResponse<R>>().await {
                        Ok(rpc_response) => {
                            if let Some(error) = rpc_response.error {
                                // RPC-level errors are not retried — they are deterministic
                                self.circuit_breaker.write().await.record_success();
                                return Err(RpcError::RpcError {
                                    code: error.code,
                                    message: error.message,
                                });
                            }

                            if let Some(result) = rpc_response.result {
                                *self.last_success.write().await = Some(Instant::now());
                                self.circuit_breaker.write().await.record_success();
                                return Ok(result);
                            }

                            return Err(RpcError::InvalidResponse("No result in response".to_string()));
                        }
                        Err(e) => {
                            last_error = Some(RpcError::InvalidResponse(e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(e.into());
                }
            }
        }

        // All retries exhausted — record failure in circuit breaker
        self.circuit_breaker.write().await.record_failure();

        Err(last_error.unwrap_or_else(|| RpcError::ConnectionFailed("Unknown error".to_string())))
    }

    // ==================== Health & Status ====================

    /// Health check
    pub async fn health_check(&self) -> Result<HealthStatus, RpcError> {
        self.send_request("helix_health", ()).await
    }

    /// Get node capabilities
    pub async fn get_capabilities(&self) -> Result<NodeCapabilities, RpcError> {
        self.send_request("helix_capabilities", ()).await
    }

    /// Get network status
    pub async fn get_network_status(&self) -> Result<NetworkStatus, RpcError> {
        self.send_request("helix_networkStatus", ()).await
    }

    // ==================== Training Operations ====================

    /// Get current training status
    pub async fn get_training_status(&self) -> Result<TrainingStatus, RpcError> {
        self.send_request("helix_getTrainingStatus", ()).await
    }

    /// Get training result for a completed round (for proof generation).
    pub async fn get_training_result(&self, round_id: u64) -> Result<TrainingResultData, RpcError> {
        #[derive(Serialize)]
        struct Params {
            round_id: u64,
        }

        self.send_request("helix_getTrainingResult", Params { round_id }).await
    }

    /// Start training for a model
    pub async fn start_training(
        &self,
        model_id: u64,
        rounds: u64,
        round_duration_secs: u64,
    ) -> Result<(), RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
            rounds: u64,
            round_duration_secs: u64,
        }

        // Void method — discard the response value
        let _: serde_json::Value = self.send_request(
            "helix_startTraining",
            Params {
                model_id,
                rounds,
                round_duration_secs,
            },
        )
        .await?;
        Ok(())
    }

    /// Stop training
    pub async fn stop_training(&self, model_id: u64) -> Result<(), RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        // Void method — discard the response value
        let _: serde_json::Value = self.send_request("helix_stopTraining", Params { model_id }).await?;
        Ok(())
    }

    /// Get training progress history
    pub async fn get_training_progress(
        &self,
        model_id: u64,
        from_round: u64,
    ) -> Result<Vec<TrainingProgress>, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
            from_round: u64,
        }

        self.send_request("helix_getTrainingProgress", Params { model_id, from_round }).await
    }

    // ==================== Proof Operations ====================

    /// Get current proof generation status
    pub async fn get_proof_status(&self) -> Result<ProofStatus, RpcError> {
        self.send_request("helix_getProofStatus", ()).await
    }

    /// Generate proof for current training step
    pub async fn generate_proof(&self, model_id: u64, round_id: u64) -> Result<GenerateProofAck, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
            round_id: u64,
        }

        self.send_request("helix_generateProof", Params { model_id, round_id }).await
    }

    /// Submit proof to blockchain
    pub async fn submit_proof(&self, submission: &ProofSubmission) -> Result<String, RpcError> {
        self.send_request("helix_submitProof", submission).await
    }

    /// Verify a proof
    pub async fn verify_proof(&self, submission: &ProofSubmission) -> Result<bool, RpcError> {
        self.send_request("helix_verifyProof", submission).await
    }

    // ==================== Model Operations ====================

    /// Get model information
    pub async fn get_model(&self, model_id: u64) -> Result<ModelInfo, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        self.send_request("helix_getModelState", Params { model_id }).await
    }

    /// List all models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, RpcError> {
        self.send_request("helix_listModels", ()).await
    }

    /// Register a new model
    pub async fn register_model(
        &self,
        name: &str,
        ipfs_hash: &str,
        min_stake: f64,
    ) -> Result<u64, RpcError> {
        #[derive(Serialize)]
        struct Params<'a> {
            name: &'a str,
            ipfs_hash: &'a str,
            min_stake: f64,
        }

        self.send_request(
            "helix_registerModel",
            Params {
                name,
                ipfs_hash,
                min_stake,
            },
        )
        .await
    }

    // ==================== Round Operations ====================

    /// Get current round information
    pub async fn get_current_round(&self, model_id: u64) -> Result<RoundInfo, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        self.send_request("helix_getCurrentRound", Params { model_id }).await
    }

    /// Get round by ID
    pub async fn get_round(&self, model_id: u64, round_id: u64) -> Result<RoundInfo, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
            round_id: u64,
        }

        self.send_request("helix_getRound", Params { model_id, round_id }).await
    }

    // ==================== Worker Operations ====================

    /// Get all workers
    pub async fn get_workers(&self) -> Result<Vec<WorkerInfo>, RpcError> {
        self.send_request("helix_getWorkers", ()).await
    }

    /// Get specific worker
    pub async fn get_worker(&self, worker_id: &str) -> Result<WorkerInfo, RpcError> {
        #[derive(Serialize)]
        struct Params<'a> {
            worker_id: &'a str,
        }

        self.send_request("helix_getWorker", Params { worker_id }).await
    }

    /// Get this node's worker info
    pub async fn get_self_worker(&self) -> Result<WorkerInfo, RpcError> {
        self.send_request("helix_getSelfWorker", ()).await
    }

    // ==================== Staking Operations ====================

    /// Get staking information
    pub async fn get_staking_info(&self, model_id: u64) -> Result<StakingInfo, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        self.send_request("helix_getStakingInfo", Params { model_id }).await
    }

    /// Stake tokens for a model
    pub async fn stake(&self, model_id: u64, amount_eth: f64) -> Result<String, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
            amount_eth: f64,
        }

        self.send_request("helix_stake", Params { model_id, amount_eth }).await
    }

    /// Unstake tokens
    pub async fn unstake(&self, model_id: u64) -> Result<String, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        self.send_request("helix_unstake", Params { model_id }).await
    }

    /// Claim accumulated rewards
    pub async fn claim_rewards(&self, model_id: u64) -> Result<String, RpcError> {
        #[derive(Serialize)]
        struct Params {
            model_id: u64,
        }

        self.send_request("helix_claimRewards", Params { model_id }).await
    }

    // ==================== Subscription Helpers ====================

    /// Poll for training status updates
    pub async fn subscribe_training_status(
        &self,
        callback: impl Fn(TrainingStatus) + Send + Sync + 'static,
        poll_interval: Duration,
        mut stop_rx: tokio::sync::watch::Receiver<bool>,
    ) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll_interval) => {
                    if let Ok(status) = self.get_training_status().await {
                        callback(status);
                    }
                }
                _ = stop_rx.changed() => {
                    if *stop_rx.borrow() {
                        break;
                    }
                }
            }
        }
    }

    /// Poll for proof status updates
    pub async fn subscribe_proof_status(
        &self,
        callback: impl Fn(ProofStatus) + Send + Sync + 'static,
        poll_interval: Duration,
        mut stop_rx: tokio::sync::watch::Receiver<bool>,
    ) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll_interval) => {
                    if let Ok(status) = self.get_proof_status().await {
                        callback(status);
                    }
                }
                _ = stop_rx.changed() => {
                    if *stop_rx.borrow() {
                        break;
                    }
                }
            }
        }
    }

    // ==================== Convenience Methods ====================

    /// Get complete demo status snapshot
    pub async fn get_demo_snapshot(&self) -> Result<DemoSnapshot, RpcError> {
        let training = self.get_training_status().await.ok();
        let proof = self.get_proof_status().await.ok();
        let network = self.get_network_status().await.ok();
        let workers = self.get_workers().await.ok();
        let health = self.health_check().await.ok();

        Ok(DemoSnapshot {
            training,
            proof,
            network,
            workers,
            health,
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
}

/// Complete status snapshot for demos
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoSnapshot {
    pub training: Option<TrainingStatus>,
    pub proof: Option<ProofStatus>,
    pub network: Option<NetworkStatus>,
    pub workers: Option<Vec<WorkerInfo>>,
    pub health: Option<HealthStatus>,
    pub timestamp: i64,
}

impl Clone for HelixRpcClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            request_id: self.request_id.clone(),
            connected: self.connected.clone(),
            last_success: self.last_success.clone(),
            capabilities: self.capabilities.clone(),
            circuit_breaker: self.circuit_breaker.clone(),
        }
    }
}

/// Mock RPC client for testing/demo without real node
pub struct MockRpcClient {
    /// Simulated training status
    training_status: Arc<RwLock<TrainingStatus>>,
    /// Simulated proof status
    proof_status: Arc<RwLock<ProofStatus>>,
    /// Simulated workers
    workers: Arc<RwLock<Vec<WorkerInfo>>>,
    /// Training progress history
    progress_history: Arc<RwLock<Vec<TrainingProgress>>>,
    /// Auto-incrementing model ID counter
    next_model_id: Arc<std::sync::atomic::AtomicU64>,
}

impl MockRpcClient {
    pub fn new() -> Self {
        Self {
            training_status: Arc::new(RwLock::new(TrainingStatus {
                active: false,
                phase: TrainingPhase::Idle,
                current_round: 0,
                total_rounds: 10,
                current_loss: 2.5,
                accumulated_error: 0.0,
                max_error_bound: 1000.0,
                round_elapsed_ms: 0,
                estimated_remaining_ms: 0,
                model_id: 0,
                started_at: 0,
            })),
            proof_status: Arc::new(RwLock::new(ProofStatus {
                generating: false,
                phase: ProofPhase::Idle,
                progress_percent: 0,
                constraints_satisfied: 0,
                total_constraints: 0,
                elapsed_ms: 0,
                estimated_remaining_ms: 0,
                memory_usage_bytes: 0,
                gpu_accelerated: false,
                error_bound: 0.0,
            })),
            workers: Arc::new(RwLock::new(Vec::new())),
            progress_history: Arc::new(RwLock::new(Vec::new())),
            next_model_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }

    /// Initialize mock workers
    pub async fn init_workers(&self, count: u32) {
        let mut workers = self.workers.write().await;
        workers.clear();
        for i in 0..count {
            workers.push(WorkerInfo {
                id: format!("worker-{}", i + 1),
                address: format!("0x{:040x}", 0x742d35Cc6634C053u64 + i as u64),
                status: WorkerStatus::Idle,
                stake: 1.0,
                proofs_submitted: 0,
                proofs_verified: 0,
                proofs_rejected: 0,
                reputation: 1.0,
                is_training: false,
                assigned_model: Some(0),
                last_activity: chrono::Utc::now().timestamp(),
            });
        }
    }

    /// Start simulated training
    pub async fn start_training(&self, rounds: u64) {
        let mut status = self.training_status.write().await;
        status.active = true;
        status.phase = TrainingPhase::Initializing;
        status.current_round = 0;
        status.total_rounds = rounds;
        status.current_loss = 2.5;
        status.accumulated_error = 0.0;
        status.started_at = chrono::Utc::now().timestamp();

        let mut workers = self.workers.write().await;
        for worker in workers.iter_mut() {
            worker.status = WorkerStatus::Training;
            worker.is_training = true;
        }
    }

    /// Advance training by one round
    pub async fn advance_round(&self) -> bool {
        let mut status = self.training_status.write().await;

        if !status.active || status.current_round >= status.total_rounds {
            status.active = false;
            status.phase = TrainingPhase::Idle;
            return false;
        }

        status.current_round += 1;
        let loss_reduction = 0.08 + (rand::random::<f64>() * 0.04);
        status.current_loss = (status.current_loss - loss_reduction).max(0.01);

        let error_increase = 3.0 + rand::random::<f64>() * 2.0;
        status.accumulated_error += error_increase;

        // Record progress
        let mut history = self.progress_history.write().await;
        history.push(TrainingProgress {
            round: status.current_round,
            loss: status.current_loss,
            error_bound: status.accumulated_error,
            timestamp: chrono::Utc::now().timestamp(),
            proof_time_ms: 200 + (rand::random::<u64>() % 100),
            verify_time_ms: 50 + (rand::random::<u64>() % 30),
            learning_rate: 0.001,
            gradient_norm: 0.5 + rand::random::<f64>() * 0.3,
        });

        // Update workers
        let mut workers = self.workers.write().await;
        for worker in workers.iter_mut() {
            worker.proofs_submitted += 1;
            worker.proofs_verified += 1;
            worker.last_activity = chrono::Utc::now().timestamp();
        }

        true
    }

    /// Get current status
    pub async fn get_training_status(&self) -> TrainingStatus {
        self.training_status.read().await.clone()
    }

    /// Get proof status
    pub async fn get_proof_status(&self) -> ProofStatus {
        self.proof_status.read().await.clone()
    }

    /// Get workers
    pub async fn get_workers(&self) -> Vec<WorkerInfo> {
        self.workers.read().await.clone()
    }

    /// Get progress history
    pub async fn get_progress_history(&self) -> Vec<TrainingProgress> {
        self.progress_history.read().await.clone()
    }

    /// Set training phase
    pub async fn set_phase(&self, phase: TrainingPhase) {
        let mut status = self.training_status.write().await;
        status.phase = phase;
    }

    /// Set proof phase
    pub async fn set_proof_phase(&self, phase: ProofPhase, progress: u8) {
        let mut status = self.proof_status.write().await;
        status.phase = phase;
        status.progress_percent = progress;
        status.generating = phase != ProofPhase::Idle && phase != ProofPhase::Complete;
    }

    /// Slash a worker
    pub async fn slash_worker(&self, worker_idx: usize) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_idx) {
            worker.status = WorkerStatus::Slashed;
            worker.stake = 0.0;
            worker.reputation = 0.0;
            worker.is_training = false;
            worker.proofs_rejected += 1;
        }
    }

    /// Fail a worker
    pub async fn fail_worker(&self, worker_idx: usize) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_idx) {
            worker.status = WorkerStatus::Faulted;
            worker.is_training = false;
        }
    }

    /// Recover a worker
    pub async fn recover_worker(&self, worker_idx: usize) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_idx) {
            if worker.status == WorkerStatus::Faulted || worker.status == WorkerStatus::Offline {
                worker.status = WorkerStatus::Training;
                worker.is_training = true;
            }
        }
    }
}

impl Default for MockRpcClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MockRpcClient {
    fn clone(&self) -> Self {
        Self {
            training_status: self.training_status.clone(),
            proof_status: self.proof_status.clone(),
            workers: self.workers.clone(),
            progress_history: self.progress_history.clone(),
            next_model_id: Arc::new(std::sync::atomic::AtomicU64::new(
                self.next_model_id.load(std::sync::atomic::Ordering::Relaxed),
            )),
        }
    }
}

// ============================================================================
// Unified RPC Client (Real + Mock)
// ============================================================================

/// Unified RPC client that can operate in real or mock mode
pub struct UnifiedRpcClient {
    /// Real RPC client (when connected to actual node)
    real_client: Option<HelixRpcClient>,
    /// Mock RPC client (for demos without real node)
    mock_client: MockRpcClient,
    /// Whether to use mock mode
    use_mock: bool,
    /// Connection status
    connected: bool,
    /// On-chain client for contract interactions (chain feature).
    /// Wrapped in Arc for Clone support.
    #[cfg(feature = "chain")]
    chain_client: Option<Arc<super::chain::ChainClient>>,
}

impl UnifiedRpcClient {
    /// Connect to a real HELIX node. Returns an error on connection failure.
    ///
    /// In production code, always use this method so misconfigurations are
    /// caught immediately instead of silently running in mock mode.
    pub async fn connect(config: HelixRpcConfig) -> Result<Self, RpcError> {
        let client = HelixRpcClient::new(config.clone())?;
        client.connect().await?;

        tracing::info!("Connected to HELIX node at {}", config.endpoint);
        Ok(Self {
            real_client: Some(client),
            mock_client: MockRpcClient::new(),
            use_mock: false,
            connected: true,
            #[cfg(feature = "chain")]
            chain_client: None,
        })
    }

    /// Create a disconnected client that is NOT in mock mode.
    ///
    /// RPC calls will return `NodeUnavailable` errors until [`connect`] is
    /// called. This is the correct default for production code — misconfigurations
    /// surface as errors instead of silently running in mock mode.
    pub fn new_disconnected() -> Self {
        Self {
            real_client: None,
            mock_client: MockRpcClient::new(),
            use_mock: false,
            connected: false,
            #[cfg(feature = "chain")]
            chain_client: None,
        }
    }

    /// Create in mock-only mode (for demos and tests).
    ///
    /// Use this when you explicitly want a mock client for development or
    /// testing. Unlike [`connect`](Self::connect), this never attempts a
    /// real connection.
    pub fn new_mock() -> Self {
        Self {
            real_client: None,
            mock_client: MockRpcClient::new(),
            use_mock: true,
            connected: false,
            #[cfg(feature = "chain")]
            chain_client: None,
        }
    }

    /// Deprecated alias for [`new_mock`](Self::new_mock).
    #[deprecated(note = "use `new_mock()` instead")]
    pub fn mock_only() -> Self {
        Self::new_mock()
    }

    /// Attach an on-chain client for contract interactions.
    #[cfg(feature = "chain")]
    pub fn set_chain_client(&mut self, client: super::chain::ChainClient) {
        self.chain_client = Some(Arc::new(client));
    }

    /// Get a reference to the on-chain client, if attached.
    #[cfg(feature = "chain")]
    pub fn chain_client(&self) -> Option<&super::chain::ChainClient> {
        self.chain_client.as_deref()
    }

    /// Check if using mock mode
    pub fn is_mock(&self) -> bool {
        self.use_mock
    }

    /// Check if connected to real node
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Initialize workers (mock or signal to real node)
    pub async fn init_workers(&self, count: u32) {
        if self.use_mock {
            self.mock_client.init_workers(count).await;
        }
        // Real node manages its own workers
    }

    /// Start training
    pub async fn start_training(&self, rounds: u64) -> Result<(), RpcError> {
        if self.use_mock {
            self.mock_client.start_training(rounds).await;
            Ok(())
        } else if let Some(ref client) = self.real_client {
            client.start_training(0, rounds, 60).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get training status
    pub async fn get_training_status(&self) -> Result<TrainingStatus, RpcError> {
        if self.use_mock {
            Ok(self.mock_client.get_training_status().await)
        } else if let Some(ref client) = self.real_client {
            client.get_training_status().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get training result for a completed round (for proof generation).
    pub async fn get_training_result(&self, round_id: u64) -> Result<TrainingResultData, RpcError> {
        if self.use_mock {
            Ok(TrainingResultData {
                available: true,
                round_id,
                worker_count: 1,
                loss: 0.0,
                error_bound: 0.0,
                model_dims: None,
                step_number: round_id,
            })
        } else if let Some(ref client) = self.real_client {
            client.get_training_result(round_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get proof status
    pub async fn get_proof_status(&self) -> Result<ProofStatus, RpcError> {
        if self.use_mock {
            Ok(self.mock_client.get_proof_status().await)
        } else if let Some(ref client) = self.real_client {
            client.get_proof_status().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get workers
    pub async fn get_workers(&self) -> Result<Vec<WorkerInfo>, RpcError> {
        if self.use_mock {
            Ok(self.mock_client.get_workers().await)
        } else if let Some(ref client) = self.real_client {
            client.get_workers().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Advance training round (mock mode)
    pub async fn advance_round(&self) -> bool {
        if self.use_mock {
            self.mock_client.advance_round().await
        } else {
            true // Real training advances automatically
        }
    }

    /// Set training phase (mock mode)
    pub async fn set_phase(&self, phase: TrainingPhase) {
        if self.use_mock {
            self.mock_client.set_phase(phase).await;
        }
    }

    /// Set proof phase (mock mode)
    pub async fn set_proof_phase(&self, phase: ProofPhase, progress: u8) {
        if self.use_mock {
            self.mock_client.set_proof_phase(phase, progress).await;
        }
    }

    /// Slash a worker (mock mode or real)
    pub async fn slash_worker(&self, worker_idx: usize) {
        if self.use_mock {
            self.mock_client.slash_worker(worker_idx).await;
        }
        // Real slashing happens on-chain
    }

    /// Get network status
    pub async fn get_network_status(&self) -> Result<NetworkStatus, RpcError> {
        if self.use_mock {
            Ok(NetworkStatus {
                peer_count: 5,
                active_workers: 3,
                active_aggregators: 2,
                avg_latency_ms: 25,
                block_height: 12345678,
                chain_id: 31337,
                blockchain_connected: true,
                coordinator_address: "0x5FbDB2315678afecb367f032d93F642f64180aa3".into(),
                bandwidth_bps: 1024 * 1024,
            })
        } else if let Some(ref client) = self.real_client {
            client.get_network_status().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get staking info
    pub async fn get_staking_info(&self, model_id: u64) -> Result<StakingInfo, RpcError> {
        if self.use_mock {
            Ok(StakingInfo {
                total_staked: 5.0,
                your_stake: 1.0,
                lock_until: chrono::Utc::now().timestamp() + 86400 * 7,
                pending_rewards: 0.05,
                total_rewards_claimed: 0.0,
                is_locked: true,
                slashing_events: Vec::new(),
            })
        } else if let Some(ref client) = self.real_client {
            client.get_staking_info(model_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get progress history
    pub async fn get_progress_history(&self) -> Vec<TrainingProgress> {
        if self.use_mock {
            self.mock_client.get_progress_history().await
        } else {
            Vec::new() // Real client would call get_training_progress
        }
    }

    /// Get complete demo snapshot
    pub async fn get_demo_snapshot(&self) -> Result<DemoSnapshot, RpcError> {
        if self.use_mock {
            Ok(DemoSnapshot {
                training: Some(self.mock_client.get_training_status().await),
                proof: Some(self.mock_client.get_proof_status().await),
                network: self.get_network_status().await.ok(),
                workers: Some(self.mock_client.get_workers().await),
                health: Some(HealthStatus {
                    healthy: true,
                    components: HashMap::new(),
                    last_check: chrono::Utc::now().timestamp(),
                    uptime_secs: 3600,
                    version: "0.1.0".into(),
                }),
                timestamp: chrono::Utc::now().timestamp(),
            })
        } else if let Some(ref client) = self.real_client {
            client.get_demo_snapshot().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get model information by ID.
    pub async fn get_model(&self, model_id: u64) -> Result<ModelInfo, RpcError> {
        if self.use_mock {
            Ok(ModelInfo {
                id: model_id,
                name: format!("helix-model-{}", model_id),
                ipfs_hash: "QmXoYP...mock".to_string(),
                architecture: "MLP".to_string(),
                parameter_count: 4096,
                current_commitment: format!("0x{}", "ab".repeat(32)),
                owner: format!("0x{}", "42".repeat(20)),
                min_stake: 0.1,
                training_active: true,
                current_round: 42,
                accumulated_error: 45.2,
                created_at: chrono::Utc::now().timestamp() - 86400,
            })
        } else if let Some(ref client) = self.real_client {
            client.get_model(model_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get round information by model and round ID.
    pub async fn get_round(&self, model_id: u64, round_id: u64) -> Result<RoundInfo, RpcError> {
        if self.use_mock {
            Ok(RoundInfo {
                round_id,
                model_id,
                started_at: chrono::Utc::now().timestamp() - 300,
                deadline: chrono::Utc::now().timestamp() + 300,
                completed: false,
                proofs_submitted: 3,
                proofs_verified: 2,
                participants: vec![
                    format!("0x{}", "11".repeat(20)),
                    format!("0x{}", "22".repeat(20)),
                ],
                prev_commitment: format!("0x{}", "ab".repeat(32)),
                new_commitment: None,
                loss: Some(0.234),
                error_delta: Some(5.1),
            })
        } else if let Some(ref client) = self.real_client {
            client.get_round(model_id, round_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get the current round for a model.
    pub async fn get_current_round(&self, model_id: u64) -> Result<RoundInfo, RpcError> {
        if self.use_mock {
            self.get_round(model_id, 0).await
        } else if let Some(ref client) = self.real_client {
            client.get_current_round(model_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Get the mock client for direct access (when in mock mode)
    pub fn mock(&self) -> &MockRpcClient {
        &self.mock_client
    }

    // ==================== SDK Forwarding Methods ====================

    /// List all registered models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, RpcError> {
        if self.use_mock {
            Ok(vec![ModelInfo {
                id: 0,
                name: "helix-demo-model".to_string(),
                ipfs_hash: "QmXoYP...mock".to_string(),
                architecture: "MLP".to_string(),
                parameter_count: 4096,
                current_commitment: format!("0x{}", "ab".repeat(32)),
                owner: format!("0x{}", "42".repeat(20)),
                min_stake: 0.1,
                training_active: false,
                current_round: 0,
                accumulated_error: 0.0,
                created_at: chrono::Utc::now().timestamp(),
            }])
        } else if let Some(ref client) = self.real_client {
            client.list_models().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Register a new model.
    pub async fn register_model(
        &self,
        name: &str,
        ipfs_hash: &str,
        min_stake: f64,
    ) -> Result<u64, RpcError> {
        if self.use_mock {
            Ok(self
                .mock_client
                .next_model_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        } else if let Some(ref client) = self.real_client {
            client.register_model(name, ipfs_hash, min_stake).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Stake tokens for a model.
    pub async fn stake(&self, model_id: u64, amount_eth: f64) -> Result<String, RpcError> {
        if self.use_mock {
            Ok(format!("0xmock_stake_tx_{}", model_id))
        } else if let Some(ref client) = self.real_client {
            client.stake(model_id, amount_eth).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Unstake tokens for a model.
    pub async fn unstake(&self, model_id: u64) -> Result<String, RpcError> {
        if self.use_mock {
            Ok(format!("0xmock_unstake_tx_{}", model_id))
        } else if let Some(ref client) = self.real_client {
            client.unstake(model_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Claim rewards for a model.
    pub async fn claim_rewards(&self, model_id: u64) -> Result<String, RpcError> {
        if self.use_mock {
            Ok(format!("0xmock_rewards_tx_{}", model_id))
        } else if let Some(ref client) = self.real_client {
            client.claim_rewards(model_id).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Health check.
    pub async fn health_check(&self) -> Result<HealthStatus, RpcError> {
        if self.use_mock {
            Ok(HealthStatus {
                healthy: true,
                components: HashMap::new(),
                last_check: chrono::Utc::now().timestamp(),
                uptime_secs: 3600,
                version: env!("CARGO_PKG_VERSION").to_string(),
            })
        } else if let Some(ref client) = self.real_client {
            client.health_check().await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }

    /// Start training for a specific model with full parameters.
    pub async fn start_training_for(
        &self,
        model_id: u64,
        rounds: u64,
        round_duration_secs: u64,
    ) -> Result<(), RpcError> {
        if self.use_mock {
            // Update the model_id on the mock status
            {
                let mut status = self.mock_client.training_status.write().await;
                status.model_id = model_id;
            }
            self.mock_client.start_training(rounds).await;
            Ok(())
        } else if let Some(ref client) = self.real_client {
            client.start_training(model_id, rounds, round_duration_secs).await
        } else {
            Err(RpcError::NodeUnavailable("No client available".into()))
        }
    }
}

impl Clone for UnifiedRpcClient {
    fn clone(&self) -> Self {
        Self {
            real_client: self.real_client.clone(),
            mock_client: self.mock_client.clone(),
            use_mock: self.use_mock,
            connected: self.connected,
            #[cfg(feature = "chain")]
            chain_client: self.chain_client.clone(),
        }
    }
}

impl Default for UnifiedRpcClient {
    fn default() -> Self {
        Self::new_disconnected()
    }
}

// ============================================================================
// On-Chain Integration (chain feature)
// ============================================================================

#[cfg(feature = "chain")]
impl UnifiedRpcClient {
    /// Submit a proof to the coordinator contract.
    ///
    /// Returns the transaction receipt or an error if no chain client is attached.
    pub async fn submit_proof_onchain(
        &self,
        model_id: u64,
        round_id: u64,
        proof: Vec<u8>,
        inputs: &super::chain::TrainingProofInputs,
    ) -> anyhow::Result<ethers::types::TransactionReceipt> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached — configure with --rpc-url and --private-key"))?;
        chain.submit_proof(model_id, round_id, proof, inputs).await
    }

    /// Get model state from the coordinator contract.
    pub async fn get_model_state_onchain(
        &self,
        model_id: u64,
    ) -> anyhow::Result<super::chain::ChainModelState> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached"))?;
        chain.get_model_state(model_id).await
    }

    /// Get round state from the coordinator contract.
    pub async fn get_round_state_onchain(
        &self,
        model_id: u64,
        round_id: u64,
    ) -> anyhow::Result<super::chain::ChainRoundState> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached"))?;
        chain.get_round_state(model_id, round_id).await
    }

    /// Stake ETH for a model via the coordinator contract.
    pub async fn stake_onchain(
        &self,
        model_id: u64,
        amount: ethers::types::U256,
    ) -> anyhow::Result<ethers::types::TransactionReceipt> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached"))?;
        chain.stake(model_id, amount).await
    }

    /// Register a model via the coordinator contract.
    pub async fn register_model_onchain(
        &self,
        ipfs_hash: &str,
        initial_commitment: ethers::types::U256,
        min_stake: ethers::types::U256,
    ) -> anyhow::Result<(ethers::types::TransactionReceipt, u64)> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached"))?;
        chain.register_model(ipfs_hash, initial_commitment, min_stake).await
    }

    /// Start a round via the coordinator contract.
    pub async fn start_round_onchain(
        &self,
        model_id: u64,
        duration_secs: u64,
    ) -> anyhow::Result<ethers::types::TransactionReceipt> {
        let chain = self
            .chain_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No chain client attached"))?;
        chain.start_round(model_id, duration_secs).await
    }
}

// ============================================================================
// Real-Time Status Tracking
// ============================================================================

/// Real-time proof generation tracker with callback support
pub struct RealTimeProofTracker {
    /// Status receiver
    status: Arc<RwLock<ProofStatus>>,
    /// Progress callbacks
    callbacks: Arc<RwLock<Vec<Box<dyn Fn(&ProofStatus) + Send + Sync>>>>,
    /// Last update timestamp
    last_update: Arc<RwLock<Instant>>,
}

impl RealTimeProofTracker {
    pub fn new() -> Self {
        Self {
            status: Arc::new(RwLock::new(ProofStatus {
                generating: false,
                phase: ProofPhase::Idle,
                progress_percent: 0,
                constraints_satisfied: 0,
                total_constraints: 0,
                elapsed_ms: 0,
                estimated_remaining_ms: 0,
                memory_usage_bytes: 0,
                gpu_accelerated: false,
                error_bound: 0.0,
            })),
            callbacks: Arc::new(RwLock::new(Vec::new())),
            last_update: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Register a progress callback
    pub async fn on_progress(&self, callback: impl Fn(&ProofStatus) + Send + Sync + 'static) {
        self.callbacks.write().await.push(Box::new(callback));
    }

    /// Update status and notify callbacks
    pub async fn update(&self, status: ProofStatus) {
        *self.last_update.write().await = Instant::now();
        *self.status.write().await = status.clone();

        let callbacks = self.callbacks.read().await;
        for callback in callbacks.iter() {
            callback(&status);
        }
    }

    /// Get current status
    pub async fn get_status(&self) -> ProofStatus {
        self.status.read().await.clone()
    }

    /// Get time since last update
    pub async fn time_since_update(&self) -> Duration {
        self.last_update.read().await.elapsed()
    }
}

impl Default for RealTimeProofTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time training status tracker
pub struct RealTimeTrainingTracker {
    /// Status
    status: Arc<RwLock<TrainingStatus>>,
    /// Loss history
    loss_history: Arc<RwLock<Vec<(u64, f64)>>>,
    /// Error bound history
    error_history: Arc<RwLock<Vec<(u64, f64)>>>,
    /// Start time
    start_time: Arc<RwLock<Option<Instant>>>,
}

impl RealTimeTrainingTracker {
    pub fn new() -> Self {
        Self {
            status: Arc::new(RwLock::new(TrainingStatus {
                active: false,
                phase: TrainingPhase::Idle,
                current_round: 0,
                total_rounds: 0,
                current_loss: 0.0,
                accumulated_error: 0.0,
                max_error_bound: 1000.0,
                round_elapsed_ms: 0,
                estimated_remaining_ms: 0,
                model_id: 0,
                started_at: 0,
            })),
            loss_history: Arc::new(RwLock::new(Vec::new())),
            error_history: Arc::new(RwLock::new(Vec::new())),
            start_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Start tracking
    pub async fn start(&self, total_rounds: u64) {
        let mut status = self.status.write().await;
        status.active = true;
        status.total_rounds = total_rounds;
        status.started_at = chrono::Utc::now().timestamp();
        *self.start_time.write().await = Some(Instant::now());
    }

    /// Update round progress
    pub async fn update_round(&self, round: u64, loss: f64, error_bound: f64, phase: TrainingPhase) {
        let mut status = self.status.write().await;
        status.current_round = round;
        status.current_loss = loss;
        status.accumulated_error = error_bound;
        status.phase = phase;

        if let Some(start) = *self.start_time.read().await {
            status.round_elapsed_ms = start.elapsed().as_millis() as u64;

            // Estimate remaining time
            if round > 0 {
                let avg_round_time = status.round_elapsed_ms / round;
                let remaining_rounds = status.total_rounds - round;
                status.estimated_remaining_ms = avg_round_time * remaining_rounds;
            }
        }

        // Record history
        self.loss_history.write().await.push((round, loss));
        self.error_history.write().await.push((round, error_bound));
    }

    /// Get current status
    pub async fn get_status(&self) -> TrainingStatus {
        self.status.read().await.clone()
    }

    /// Get loss history
    pub async fn get_loss_history(&self) -> Vec<(u64, f64)> {
        self.loss_history.read().await.clone()
    }

    /// Get error history
    pub async fn get_error_history(&self) -> Vec<(u64, f64)> {
        self.error_history.read().await.clone()
    }

    /// Get elapsed time
    pub async fn elapsed(&self) -> Option<Duration> {
        self.start_time.read().await.map(|s| s.elapsed())
    }
}

impl Default for RealTimeTrainingTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HelixRpcConfig::default();
        assert_eq!(config.endpoint, "http://127.0.0.1:9002/rpc");
        assert_eq!(config.timeout_secs, 30);
    }

    #[tokio::test]
    async fn test_mock_client() {
        let mock = MockRpcClient::new();
        mock.init_workers(3).await;

        let workers = mock.get_workers().await;
        assert_eq!(workers.len(), 3);

        mock.start_training(5).await;

        let status = mock.get_training_status().await;
        assert!(status.active);
        assert_eq!(status.total_rounds, 5);

        // Advance a round
        mock.advance_round().await;

        let status = mock.get_training_status().await;
        assert_eq!(status.current_round, 1);
        assert!(status.current_loss < 2.5);
    }

    #[test]
    fn test_training_phase_names() {
        assert_eq!(TrainingPhase::Forward.name(), "Forward Pass");
        assert_eq!(TrainingPhase::ProofGeneration.name(), "Generating Proof");
    }

    #[test]
    fn test_proof_phase_names() {
        assert_eq!(ProofPhase::WitnessGeneration.name(), "Generating Witness");
        assert_eq!(ProofPhase::ProofComputation.name(), "Computing Proof");
    }
}
