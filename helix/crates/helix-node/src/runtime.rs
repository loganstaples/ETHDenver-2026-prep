//! Node Runtime — manages the full lifecycle of all subsystems.
//!
//! The `NodeRuntime` boots subsystems in dependency order:
//! 1. Network (P2P transport, gossip, discovery)
//! 2. JSON-RPC server
//! 3. HTTP API server (aggregator only)
//! 4. Training orchestrator (aggregator only)
//! 5. Failure detector (aggregator only)
//! 6. On-chain pipeline (if chain config present)
//!
//! Graceful shutdown tears down in reverse order on SIGINT/SIGTERM.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{error, info, warn, instrument};
use parking_lot::RwLock;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

use crate::api::http::{ApiRateLimiter, ApiState, FaultToleranceStatus, MpcHealthStatus};
use crate::metrics::NodeMetrics;
use crate::api::rpc::{
    MPCStatusSnapshot, NodeConfigSnapshot, ProofStatusEntry, RpcRateLimiter, RpcState,
    start_rpc_server,
};
use crate::config::{NodeConfig, NodeRole};
use crate::network::messages::{
    HeartbeatMessage, MessagePayload, NodeCapabilities, PeerId, TrainingMessage, TrainingParams,
};
use crate::network::runner::{NetworkEvent, NetworkRunner, NetworkRunnerBuilder};
use crate::round_commit::{RoundCommitConfig, RoundCommitManager};
use crate::sc_client::SCClient;
use crate::trainer::{average_models, MlpModel, Trainer};
use crate::training::orchestrator::{OrchestratorConfig, OrchestratorEvent, TrainingOrchestrator};
use crate::training::fault_tolerance::{
    FailureDetector, FaultEvent, FaultToleranceConfig, WorkerHealth,
};
use crate::training::persistence::{
    AggregatorSnapshot, StatePersistence, WorkerSnapshot,
};
use crate::training::{
    MPCTrainingConfig, MPCWorkerHandle, MpcSessionOrchestrator, PrivateAggregator,
};
use crate::api::http::{OrchestratorSnapshot, PeerSnapshot, MetricsSnapshot};

use helix_core::ModelCheckpoint;

// ============================================================================
// Subsystem Health Tracking
// ============================================================================

/// Status of an individual subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubsystemStatus {
    /// Not yet started.
    Pending,
    /// Currently starting up.
    Starting,
    /// Running normally.
    Running,
    /// Failed to start or crashed during operation.
    Failed(String),
    /// Cleanly stopped during shutdown.
    Stopped,
}

impl std::fmt::Display for SubsystemStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::Failed(msg) => write!(f, "failed: {}", msg),
            Self::Stopped => write!(f, "stopped"),
        }
    }
}

/// Health status of all subsystems, exposed via RPC.
#[derive(Debug, Clone)]
pub struct RuntimeHealth {
    pub network: SubsystemStatus,
    pub rpc_server: SubsystemStatus,
    pub http_api: SubsystemStatus,
    pub training_orchestrator: SubsystemStatus,
    pub failure_detector: SubsystemStatus,
    pub chain_pipeline: SubsystemStatus,
    pub chain_watcher: SubsystemStatus,
    pub started_at: Instant,
}

impl RuntimeHealth {
    fn new() -> Self {
        Self {
            network: SubsystemStatus::Pending,
            rpc_server: SubsystemStatus::Pending,
            http_api: SubsystemStatus::Pending,
            training_orchestrator: SubsystemStatus::Pending,
            failure_detector: SubsystemStatus::Pending,
            chain_pipeline: SubsystemStatus::Pending,
            chain_watcher: SubsystemStatus::Pending,
            started_at: Instant::now(),
        }
    }

    /// Returns true if all required subsystems are running.
    pub fn is_healthy(&self) -> bool {
        matches!(self.network, SubsystemStatus::Running)
            && matches!(self.rpc_server, SubsystemStatus::Running)
    }

    /// Returns a JSON-serializable summary for the helix_health RPC.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "healthy": self.is_healthy(),
            "uptime_secs": self.started_at.elapsed().as_secs(),
            "components": {
                "network": self.network.to_string(),
                "rpc_server": self.rpc_server.to_string(),
                "http_api": self.http_api.to_string(),
                "training_orchestrator": self.training_orchestrator.to_string(),
                "failure_detector": self.failure_detector.to_string(),
                "chain_pipeline": self.chain_pipeline.to_string(),
                "chain_watcher": self.chain_watcher.to_string(),
            },
            "version": env!("CARGO_PKG_VERSION"),
        })
    }
}

// ============================================================================
// Node Runtime
// ============================================================================

/// The main node runtime that owns and coordinates all subsystems.
pub struct NodeRuntime {
    config: NodeConfig,
    health: Arc<RwLock<RuntimeHealth>>,
    shutdown_tx: watch::Sender<bool>,
    /// Node-wide atomic metrics shared across all subsystems.
    node_metrics: Arc<NodeMetrics>,
}

impl NodeRuntime {
    /// Creates a new runtime from configuration.
    pub fn new(config: NodeConfig) -> Self {
        let (shutdown_tx, _shutdown_rx) = watch::channel(false);
        Self {
            config,
            health: Arc::new(RwLock::new(RuntimeHealth::new())),
            shutdown_tx,
            node_metrics: NodeMetrics::new(),
        }
    }

    /// Returns the current health status.
    pub fn health(&self) -> RuntimeHealth {
        self.health.read().clone()
    }

    /// Returns a reference to the node-wide metrics.
    pub fn metrics(&self) -> &Arc<NodeMetrics> {
        &self.node_metrics
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &NodeConfig {
        &self.config
    }

    /// Runs the node until shutdown signal is received.
    ///
    /// Boots all subsystems in dependency order, runs the main event loop,
    /// and performs graceful shutdown on SIGINT/SIGTERM.
    #[instrument(skip_all)]
    pub async fn run(&self) -> anyhow::Result<()> {
        info!(
            "Starting HELIX node (role={}, addr={}, rpc_port={})",
            self.config.role_str(),
            self.config.listen_addr,
            self.config.rpc_port,
        );

        match self.config.role {
            NodeRole::Compute => self.run_worker().await,
            NodeRole::Aggregator => self.run_aggregator().await,
            NodeRole::Verifier => {
                // Verifier runs the same as worker but with different capabilities
                self.run_worker().await
            }
        }
    }

    /// Triggers graceful shutdown.
    #[instrument(skip_all)]
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    // ========================================================================
    // Network Setup (shared between worker and aggregator)
    // ========================================================================

    fn build_network(&self, local_id: &PeerId, is_aggregator: bool) -> anyhow::Result<NetworkRunner> {
        let listen_addr: SocketAddr = self.config.listen_addr.parse()
            .map_err(|e| anyhow::anyhow!("invalid listen_addr: {}", e))?;

        let capabilities = NodeCapabilities {
            can_train: !is_aggregator,
            can_aggregate: is_aggregator,
            can_prove: !is_aggregator,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 10,
        };

        let mut network = NetworkRunnerBuilder::new()
            .local_id(local_id.clone())
            .listen_addr(listen_addr)
            .bootstrap_nodes(self.config.bootstrap_nodes.clone())
            .enable_mdns(self.config.mdns_enabled)
            .capabilities(capabilities)
            .build()?;

        #[cfg(feature = "crypto-sign")]
        {
            let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
            network.set_signing_key(signing_key);
            info!("ed25519 identity configured");
        }

        Ok(network)
    }

    fn build_rpc_state(
        &self,
        snapshot: Arc<RwLock<OrchestratorSnapshot>>,
        round_trigger_tx: broadcast::Sender<()>,
        proof_status: Arc<RwLock<Vec<ProofStatusEntry>>>,
        round_weights: Arc<RwLock<Vec<crate::api::rpc::RoundWeightEntry>>>,
        model_store: Option<Arc<parking_lot::Mutex<crate::storage::model_store::ModelStore>>>,
    ) -> Arc<RpcState> {
        let (stop_trigger_tx, _) = broadcast::channel::<()>(16);
        Arc::new(RpcState {
            snapshot,
            round_trigger_tx,
            stop_trigger_tx,
            node_role: self.config.role_str().to_string(),
            start_time: Instant::now(),
            proof_status,
            proof_queue: Arc::new(RwLock::new(Vec::new())),
            node_config: Arc::new(RwLock::new(NodeConfigSnapshot {
                max_workers: 100,
                round_timeout_secs: self.config.training.collection_timeout_secs,
                verification_enabled: true,
            })),
            aggregation_results: Arc::new(RwLock::new(Vec::new())),
            mpc_status: Arc::new(RwLock::new(MPCStatusSnapshot {
                enabled: self.config.mpc.enabled,
                num_parties: self.config.mpc.num_parties,
                party_index: self.config.mpc.party_index,
                d_in: self.config.training.d_in,
                d_hid: self.config.training.d_hid,
                d_out: self.config.training.d_out,
                ..Default::default()
            })),
            model_id: Arc::new(RwLock::new(None)),
            rpc_addr: format!("0.0.0.0:{}", self.config.rpc_port),
            rate_limiter: Arc::new(RwLock::new(RpcRateLimiter::default())),
            round_weights,
            worker_daemon: None,
            node_metrics: self.node_metrics.clone(),
            model_store,
        })
    }

    // ========================================================================
    // Chain Watcher Setup
    // ========================================================================

    /// Starts the on-chain event watcher if chain config is present and enabled.
    ///
    /// Returns the watcher handle, event reactor, and join handle. The reactor
    /// tracks on-chain state (active models, pause status, slash events) and
    /// can be queried by other subsystems.
    async fn start_chain_watcher(
        &self,
        local_address: Option<ethers::types::Address>,
    ) -> Option<(
        Arc<crate::chain_watcher::NodeEventReactor>,
        tokio::task::JoinHandle<()>,
    )> {
        let chain_config = self.config.chain.as_ref()?;

        if !chain_config.watcher.enabled {
            info!("Chain watcher disabled in config");
            return None;
        }

        self.health.write().chain_watcher = SubsystemStatus::Starting;

        let cursor_path = self.config.data_dir.join("chain_watcher_cursor.json");

        match crate::chain_watcher::ChainWatcher::new(
            &self.config.rpc_url,
            &chain_config.coordinator_address,
            chain_config.watcher.clone(),
            cursor_path,
        ).await {
            Ok(watcher) => {
                // Create the event reactor
                let reactor = Arc::new(
                    crate::chain_watcher::NodeEventReactor::new(local_address),
                );

                // Subscribe the reactor to events
                let rx = watcher.subscribe();
                let reactor_handle = crate::chain_watcher::spawn_event_handler(
                    rx, reactor.clone(),
                );

                // Start the watcher polling loop
                let watcher_handle = watcher.start();

                // Combine both handles into one. Move `watcher` into the
                // task so the shutdown channel stays alive until the task ends.
                let combined_handle = tokio::spawn(async move {
                    let _watcher = watcher; // keep alive for shutdown_tx
                    tokio::select! {
                        _ = watcher_handle => {
                            info!("Chain watcher polling loop exited");
                        }
                        _ = reactor_handle => {
                            info!("Chain watcher event reactor exited");
                        }
                    }
                });

                self.health.write().chain_watcher = SubsystemStatus::Running;
                info!(
                    contract = %chain_config.coordinator_address,
                    poll_secs = chain_config.watcher.poll_interval_secs,
                    confirmations = chain_config.watcher.confirmation_depth,
                    "Chain watcher started"
                );

                Some((reactor, combined_handle))
            }
            Err(e) => {
                self.health.write().chain_watcher =
                    SubsystemStatus::Failed(e.to_string());
                warn!("Failed to start chain watcher: {}. Continuing without event watching.", e);
                None
            }
        }
    }

    // ========================================================================
    // Worker Role
    // ========================================================================

    #[instrument(skip_all)]
    async fn run_worker(&self) -> anyhow::Result<()> {
        let local_id = PeerId::random();
        let _t = &self.config.training;

        info!(
            "Worker {} starting (bootstrap={}, mdns={}, mpc={})",
            local_id,
            self.config.bootstrap_nodes.len(),
            self.config.mdns_enabled,
            self.config.mpc.enabled,
        );

        // --- 1. Network ---
        self.health.write().network = SubsystemStatus::Starting;
        let network = match self.build_network(&local_id, false) {
            Ok(n) => n,
            Err(e) => {
                self.health.write().network = SubsystemStatus::Failed(e.to_string());
                return Err(anyhow::anyhow!("Network failed to initialize: {}", e));
            }
        };
        let network = Arc::new(network);
        network.start().await?;
        network.bootstrap().await;
        if self.config.mdns_enabled {
            if let Err(e) = network.start_mdns().await {
                warn!("mDNS discovery failed to start: {}", e);
            }
        }
        self.health.write().network = SubsystemStatus::Running;
        info!("Network subsystem started");

        // --- 2. RPC Server ---
        self.health.write().rpc_server = SubsystemStatus::Starting;
        let (round_trigger_tx, _) = broadcast::channel::<()>(16);
        let api_snapshot = Arc::new(RwLock::new(OrchestratorSnapshot::default()));
        let proof_status = Arc::new(RwLock::new(Vec::<ProofStatusEntry>::new()));
        let round_weights = Arc::new(RwLock::new(Vec::new()));

        // --- Worker Daemon ---
        let worker_capabilities = NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 10,
        };
        let worker_daemon = Arc::new(crate::worker::WorkerDaemon::new(
            local_id.clone(),
            self.config.worker_daemon.clone(),
            worker_capabilities,
        ));

        let mut rpc_state = self.build_rpc_state(
            api_snapshot.clone(),
            round_trigger_tx.clone(),
            proof_status.clone(),
            round_weights.clone(),
            None, // Workers don't need persistent model store
        );
        // Inject worker daemon into RPC state
        if let Some(state) = Arc::get_mut(&mut rpc_state) {
            state.worker_daemon = Some(worker_daemon.clone());
        } else {
            return Err(anyhow::anyhow!("Failed to get mutable reference to RPC state - already shared"));
        }

        let rpc_state_clone = rpc_state.clone();

        let rpc_addr: SocketAddr = format!("0.0.0.0:{}", self.config.rpc_port).parse()?;
        let rpc_handle: JoinHandle<()> = tokio::spawn(async move {
            if let Err(e) = start_rpc_server(rpc_addr, rpc_state_clone).await {
                error!("JSON-RPC server error: {}", e);
            }
        });
        self.health.write().rpc_server = SubsystemStatus::Running;
        info!("JSON-RPC server listening on {}", rpc_addr);

        // --- 3. Heartbeat loop ---
        let network_hb = network.clone();
        let mut shutdown_hb = self.shutdown_tx.subscribe();
        let hb_handle = tokio::spawn(async move {
            let mut seq = 0u64;
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(5)) => {
                        network_hb.broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                            seq,
                            is_pong: false,
                            load: 30,
                        })).await;
                        seq += 1;
                    }
                    _ = shutdown_hb.changed() => break,
                }
            }
        });

        // --- 4. Chain Watcher ---
        let _chain_watcher = self.start_chain_watcher(None).await;

        // --- 5. State persistence ---
        let persistence = self.create_persistence(&format!("worker_{}", local_id));

        let mut trainer: Option<Trainer> = None;
        let mut mpc_handle: Option<MPCWorkerHandle> = None;
        let mut steps_completed = 0u64;
        let mut last_completed_round = 0u64;

        // Restore from checkpoint if available
        if let Some(ref p) = persistence {
            if let Ok(Some(snap)) = p.load_latest_worker() {
                steps_completed = snap.steps_completed;
                last_completed_round = snap.last_completed_round;
                info!(
                    "Restored worker state: steps={}, last_round={}",
                    steps_completed, last_completed_round,
                );
            }
        }

        // --- Main event loop ---
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mpc_config = &self.config.mpc;
        let mpc_party_index = mpc_config.party_index.unwrap_or(0);
        let mpc_num_parties = mpc_config.num_parties;
        let mpc_enabled = mpc_config.enabled;
        let checkpoint_interval = self.config.fault_tolerance.checkpoint_interval_steps;

        let result = tokio::select! {
            _ = shutdown_rx.changed() => {
                info!("Worker shutting down gracefully...");

                // Notify peers of departure
                network.broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                    seq: u64::MAX,
                    is_pong: false,
                    load: 0,
                })).await;

                // Save final checkpoint
                if let Some(ref p) = persistence {
                    let mut snap = WorkerSnapshot::new(
                        local_id.to_string(),
                        last_completed_round,
                        steps_completed,
                    );
                    if let Some(ref tr) = trainer {
                        let ckpt = tr.model().to_checkpoint(tr.step_count());
                        if let Ok(bytes) = ckpt.to_bytes() {
                            snap.model_checkpoint_bytes = Some(bytes);
                        }
                    }
                    if mpc_enabled {
                        snap.mpc_party_index = Some(mpc_party_index);
                    }
                    if let Err(e) = p.save_worker(&snap) {
                        error!("Failed to save worker checkpoint on shutdown: {}", e);
                    } else {
                        info!("Worker checkpoint saved (steps={})", steps_completed);
                    }
                }
                Ok(())
            }
            result = self.worker_event_loop(
                &network,
                &rpc_state,
                &proof_status,
                &persistence,
                &local_id,
                &mut trainer,
                &mut mpc_handle,
                &mut steps_completed,
                &mut last_completed_round,
                mpc_enabled,
                mpc_party_index,
                mpc_num_parties,
                checkpoint_interval,
                &worker_daemon,
            ) => result,
        };

        // Cleanup
        hb_handle.abort();
        rpc_handle.abort();
        self.health.write().network = SubsystemStatus::Stopped;
        self.health.write().rpc_server = SubsystemStatus::Stopped;
        info!("Worker shutdown complete (steps={})", steps_completed);
        result
    }

    /// Worker's main event loop — processes network events.
    #[allow(clippy::too_many_arguments)]
    async fn worker_event_loop(
        &self,
        network: &Arc<NetworkRunner>,
        rpc_state: &Arc<RpcState>,
        proof_status: &Arc<RwLock<Vec<ProofStatusEntry>>>,
        persistence: &Option<StatePersistence>,
        local_id: &PeerId,
        trainer: &mut Option<Trainer>,
        mpc_handle: &mut Option<MPCWorkerHandle>,
        steps_completed: &mut u64,
        last_completed_round: &mut u64,
        mpc_enabled: bool,
        mpc_party_index: usize,
        mpc_num_parties: usize,
        checkpoint_interval: u64,
        worker_daemon: &Arc<crate::worker::WorkerDaemon>,
    ) -> anyhow::Result<()> {
        let t = &self.config.training;

        loop {
            match network.next_event().await {
                Some(NetworkEvent::TrainingMessage { from, message }) => {
                    match message {
                        TrainingMessage::RoundStart { round_id, model_hash: _, params } => {
                            info!(
                                "RoundStart #{} from {} ({}x{}x{}, lr={})",
                                round_id, from, params.d_in, params.d_hid, params.d_out,
                                params.learning_rate,
                            );

                            // Initialize trainer if needed
                            if trainer.is_none() {
                                *trainer = Some(Trainer::new(
                                    params.d_in, params.d_hid, params.d_out,
                                    params.learning_rate, params.model_seed,
                                ));
                            }

                            if mpc_enabled {
                                self.worker_handle_mpc_round(
                                    network, rpc_state, proof_status, persistence,
                                    local_id, trainer, mpc_handle, steps_completed,
                                    last_completed_round, mpc_party_index, mpc_num_parties,
                                    checkpoint_interval, round_id, &params,
                                ).await;
                            } else {
                                self.worker_handle_regular_round(
                                    network, rpc_state, proof_status, persistence,
                                    local_id, trainer, steps_completed, last_completed_round,
                                    checkpoint_interval, round_id, &params,
                                ).await;
                            }
                        }
                        ref msg @ TrainingMessage::ModelWeights { round_id, ref checkpoint_data, weight_hash: _ }
                        | ref msg @ TrainingMessage::UpdatedWeights { round_id, ref checkpoint_data, weight_hash: _ } => {
                            info!(
                                "Received model weights for round {} ({} bytes)",
                                round_id, checkpoint_data.len(),
                            );
                            // Forward to worker daemon
                            worker_daemon.handle_training_message(&from, msg, network).await;
                            // Also update runtime's trainer (backward compat)
                            if let Ok(ckpt) = ModelCheckpoint::from_bytes(checkpoint_data) {
                                if let Ok(model) = MlpModel::from_checkpoint(&ckpt) {
                                    let lr = trainer.as_ref()
                                        .map(|tr| tr.learning_rate())
                                        .unwrap_or(t.learning_rate);
                                    *trainer = Some(Trainer::with_model(model, lr));
                                    info!("Model loaded for round {} (step={})", round_id, ckpt.step_number);
                                }
                            }
                        }
                        TrainingMessage::RoundComplete { round_id, .. } => {
                            info!("Round {} completed", round_id);
                            *last_completed_round = round_id;

                            if let Some(ref p) = persistence {
                                let mut snap = WorkerSnapshot::new(
                                    local_id.to_string(), *last_completed_round, *steps_completed,
                                );
                                if let Some(ref tr) = trainer {
                                    let ckpt = tr.model().to_checkpoint(tr.step_count());
                                    if let Ok(bytes) = ckpt.to_bytes() {
                                        snap.model_checkpoint_bytes = Some(bytes);
                                    }
                                }
                                let _ = p.save_worker(&snap);
                            }
                        }
                        _ => {}
                    }
                }
                Some(NetworkEvent::PeerDiscovered(peer)) => {
                    info!("Peer discovered: {} at {}", peer.id, peer.address);
                }
                Some(NetworkEvent::PeerDisconnected(peer_id)) => {
                    warn!("Peer disconnected: {}", peer_id);
                    if let Some(peer_info) = network.discovery().get_peer(&peer_id).await {
                        network.schedule_reconnect(
                            peer_id.clone(),
                            peer_info.address,
                            peer_info.capabilities,
                        );
                    }
                }
                Some(NetworkEvent::RoundManagementMessage { from, message }) => {
                    worker_daemon.handle_round_management(&from, &message, network).await;
                }
                Some(NetworkEvent::RegistrationMessage { from, message }) => {
                    worker_daemon.handle_registration_message(&from, &message);
                }
                Some(NetworkEvent::Error { error, .. }) => {
                    warn!("Network error: {}", error);
                }
                None => {
                    warn!("Network event stream ended");
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Handle a training round in MPC mode.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    async fn worker_handle_mpc_round(
        &self,
        network: &Arc<NetworkRunner>,
        rpc_state: &Arc<RpcState>,
        proof_status: &Arc<RwLock<Vec<ProofStatusEntry>>>,
        persistence: &Option<StatePersistence>,
        local_id: &PeerId,
        trainer: &mut Option<Trainer>,
        mpc_handle: &mut Option<MPCWorkerHandle>,
        steps_completed: &mut u64,
        last_completed_round: &mut u64,
        mpc_party_index: usize,
        mpc_num_parties: usize,
        checkpoint_interval: u64,
        round_id: u64,
        params: &TrainingParams,
    ) {
        if mpc_handle.is_none() {
            let model = MlpModel::new_random(
                params.d_in, params.d_hid, params.d_out, params.model_seed,
            );
            *mpc_handle = Some(MPCWorkerHandle::new(
                mpc_party_index, mpc_num_parties, model,
                params.learning_rate, 100.0,
            ));
        }

        let Some(mpc) = mpc_handle.as_mut() else {
            error!("MPC handle not initialized");
            return;
        };
        let (x, target) = generate_training_data(
            params.d_in, params.d_out, params.model_seed + round_id,
        );

        match mpc.compute_gradient_share(&x, &target) {
            Ok(computation) => {
                // Generate ZK proof
                let proof_bytes = if let Some(ref mut tr) = trainer {
                    match tr.train_step(&x, &target) {
                        Ok(proved) => proved.evm_bundle
                            .as_ref()
                            .map(|b| b.evm_proof.clone())
                            .unwrap_or_else(|| proved.proof_result.proof.clone()),
                        Err(e) => {
                            warn!("MPC proof generation failed: {}", e);
                            computation.gradient_commitment.to_vec()
                        }
                    }
                } else {
                    computation.gradient_commitment.to_vec()
                };

                let (hiding_commitment, nonce) =
                    crate::training::consensus::compute_hiding_gradient_commitment(
                        &computation.gradient_commitment,
                    );

                network.broadcast(MessagePayload::Gradient(
                    crate::network::messages::GradientMessage::ShareGradient {
                        round_id,
                        gradient_commitment: hiding_commitment,
                        commitment_nonce: nonce,
                        error_bound: computation.local_loss * 0.01,
                        proof: proof_bytes,
                    },
                )).await;

                *steps_completed += 1;

                self.maybe_checkpoint_worker(
                    persistence, local_id, *last_completed_round,
                    *steps_completed, trainer, Some(mpc_party_index),
                    Some(round_id), checkpoint_interval,
                );

                // Update MPC status
                {
                    let mut status = rpc_state.mpc_status.write();
                    status.enabled = true;
                    status.num_parties = mpc_num_parties;
                    status.party_index = Some(mpc_party_index);
                    status.current_step = mpc.current_step();
                }

                proof_status.write().push(ProofStatusEntry {
                    round_id,
                    status: "submitted".to_string(),
                    proofs_collected: 1,
                });
            }
            Err(e) => error!("MPC training failed for round {}: {}", round_id, e),
        }
    }

    /// Handle a training round in regular (non-MPC) mode.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all)]
    async fn worker_handle_regular_round(
        &self,
        network: &Arc<NetworkRunner>,
        _rpc_state: &Arc<RpcState>,
        proof_status: &Arc<RwLock<Vec<ProofStatusEntry>>>,
        persistence: &Option<StatePersistence>,
        local_id: &PeerId,
        trainer: &mut Option<Trainer>,
        steps_completed: &mut u64,
        last_completed_round: &mut u64,
        checkpoint_interval: u64,
        round_id: u64,
        params: &TrainingParams,
    ) {
        let Some(tr) = trainer.as_mut() else {
            error!("Trainer not initialized");
            return;
        };
        let (x, target) = generate_training_data(
            params.d_in, params.d_out, params.model_seed + round_id,
        );

        match tr.train_step(&x, &target) {
            Ok(result) => {
                let proof_bytes = result.evm_bundle
                    .as_ref()
                    .map(|b| b.evm_proof.clone())
                    .unwrap_or_else(|| result.proof_result.proof.clone());

                let (hiding_commitment, nonce) =
                    crate::training::consensus::compute_hiding_gradient_commitment(
                        &result.commitment,
                    );

                network.broadcast(MessagePayload::Gradient(
                    crate::network::messages::GradientMessage::ShareGradient {
                        round_id,
                        gradient_commitment: hiding_commitment,
                        commitment_nonce: nonce,
                        error_bound: result.loss * 0.01,
                        proof: proof_bytes,
                    },
                )).await;

                // Send updated weights
                let updated_ckpt = tr.model().to_checkpoint(tr.step_count());
                if let Ok(ckpt_bytes) = updated_ckpt.to_bytes() {
                    network.broadcast(MessagePayload::Gradient(
                        crate::network::messages::GradientMessage::WeightUpdate {
                            round_id,
                            checkpoint_data: ckpt_bytes,
                            weight_hash: tr.model().commitment(),
                        },
                    )).await;
                }

                *steps_completed += 1;

                self.maybe_checkpoint_worker(
                    persistence, local_id, *last_completed_round,
                    *steps_completed, trainer, None, Some(round_id),
                    checkpoint_interval,
                );

                proof_status.write().push(ProofStatusEntry {
                    round_id,
                    status: "submitted".to_string(),
                    proofs_collected: 1,
                });
            }
            Err(e) => error!("Training failed for round {}: {}", round_id, e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_checkpoint_worker(
        &self,
        persistence: &Option<StatePersistence>,
        local_id: &PeerId,
        last_completed_round: u64,
        steps_completed: u64,
        trainer: &mut Option<Trainer>,
        mpc_party_index: Option<usize>,
        current_round_id: Option<u64>,
        checkpoint_interval: u64,
    ) {
        if checkpoint_interval == 0 || steps_completed % checkpoint_interval != 0 {
            return;
        }
        if let Some(ref p) = persistence {
            let mut snap = WorkerSnapshot::new(
                local_id.to_string(), last_completed_round, steps_completed,
            );
            snap.mpc_party_index = mpc_party_index;
            snap.current_round_id = current_round_id;
            if let Some(ref tr) = trainer {
                let ckpt = tr.model().to_checkpoint(tr.step_count());
                if let Ok(bytes) = ckpt.to_bytes() {
                    snap.model_checkpoint_bytes = Some(bytes);
                }
            }
            if let Err(e) = p.save_worker(&snap) {
                warn!("Step checkpoint failed: {}", e);
            } else {
                info!("Worker checkpoint at step {}", steps_completed);
            }
        }
    }

    // ========================================================================
    // Aggregator Role
    // ========================================================================

    #[instrument(skip_all)]
    async fn run_aggregator(&self) -> anyhow::Result<()> {
        let t = &self.config.training;
        let local_id = PeerId::from_string("aggregator");

        info!(
            "Aggregator starting (min_workers={}, model={}x{}x{}, mpc={})",
            t.min_workers, t.d_in, t.d_hid, t.d_out, self.config.mpc.enabled,
        );

        // --- 1. Network ---
        self.health.write().network = SubsystemStatus::Starting;
        let network = match self.build_network(&local_id, true) {
            Ok(n) => n,
            Err(e) => {
                self.health.write().network = SubsystemStatus::Failed(e.to_string());
                return Err(anyhow::anyhow!("Network failed to initialize: {}", e));
            }
        };
        let network = Arc::new(network);
        network.start().await?;
        network.bootstrap().await;
        if self.config.mdns_enabled {
            if let Err(e) = network.start_mdns().await {
                warn!("mDNS discovery failed to start: {}", e);
            }
        }
        self.health.write().network = SubsystemStatus::Running;
        info!("Network subsystem started");

        // --- 2. Training Orchestrator ---
        self.health.write().training_orchestrator = SubsystemStatus::Starting;
        let orch_config = OrchestratorConfig {
            min_workers: t.min_workers,
            collection_timeout: Duration::from_secs(t.collection_timeout_secs),
            heartbeat_interval: Duration::from_secs(10),
            worker_timeout: Duration::from_secs(t.worker_timeout_secs),
            default_params: TrainingParams {
                learning_rate: t.learning_rate,
                batch_size: 1,
                local_epochs: 1,
                max_error_bound: 0.1,
                d_in: t.d_in,
                d_hid: t.d_hid,
                d_out: t.d_out,
                model_seed: t.model_seed,
                num_layers: 2,
                activation_type: 0,
            },
            ..Default::default()
        };

        let orchestrator = TrainingOrchestrator::new(
            local_id.clone(), orch_config, network.clone(),
        );
        orchestrator.set_leader(true);

        // Wire on-chain commit manager if chain config is available
        if let Some(ref chain) = self.config.chain {
            let private_key = self.config.private_key()
                .unwrap_or_default();

            if !self.config.rpc_url.is_empty() && !private_key.is_empty() {
                info!("Connecting to chain at {} (coordinator={})",
                    self.config.rpc_url, chain.coordinator_address);
                match SCClient::with_config(
                    &self.config.rpc_url, &private_key, &chain.coordinator_address,
                ).await {
                    Ok(client) => {
                        let client = Arc::new(client);
                        let commit_config = RoundCommitConfig {
                            model_id: chain.model_id.unwrap_or(0),
                            min_proofs: 1,
                            collection_timeout: Duration::from_secs(t.collection_timeout_secs),
                            ..Default::default()
                        };
                        let manager = RoundCommitManager::with_client(commit_config, client);
                        orchestrator.set_round_commit_manager(manager);
                        self.health.write().chain_pipeline = SubsystemStatus::Running;
                        info!("On-chain proof submission enabled");
                    }
                    Err(e) => {
                        self.health.write().chain_pipeline = SubsystemStatus::Failed(e.to_string());
                        warn!("Failed to connect to chain: {}. Running without on-chain submission.", e);
                    }
                }
            }
        }

        let mut event_rx = orchestrator.start().await;
        self.health.write().training_orchestrator = SubsystemStatus::Running;
        info!("Training orchestrator started");

        // --- 3. Failure Detector ---
        self.health.write().failure_detector = SubsystemStatus::Starting;
        let ft_config = FaultToleranceConfig {
            heartbeat_interval: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(self.config.fault_tolerance.heartbeat_timeout_secs),
            max_missed_heartbeats: self.config.fault_tolerance.max_missed_heartbeats,
            min_healthy_workers: t.min_workers,
            max_failures_before_exclusion: self.config.fault_tolerance.max_failures_before_exclusion,
            failure_cooldown: Duration::from_secs(self.config.fault_tolerance.failure_cooldown_secs),
            auto_recovery: self.config.fault_tolerance.auto_recovery,
            leader_failover: false,
            max_degraded_fraction: 0.5,
        };
        let failure_detector = Arc::new(FailureDetector::new(ft_config));
        let mut fault_event_rx = failure_detector.subscribe();
        let fd_handle = failure_detector.start_detection_loop();
        self.health.write().failure_detector = SubsystemStatus::Running;
        info!("Failure detector started");

        // Spawn fault event handler
        let fd_for_events = failure_detector.clone();
        let fault_tolerance_status = Arc::new(RwLock::new(FaultToleranceStatus::default()));
        let ft_status_for_events = fault_tolerance_status.clone();
        tokio::spawn(async move {
            while let Ok(event) = fault_event_rx.recv().await {
                match &event {
                    FaultEvent::WorkerFailed { peer_id, reason, failure_count } => {
                        warn!("Worker {} failed (reason={}, count={})", peer_id, reason, failure_count);
                    }
                    FaultEvent::WorkerRecovered { peer_id } => {
                        info!("Worker {} recovered", peer_id);
                    }
                    FaultEvent::InsufficientWorkers { healthy, required } => {
                        error!("Insufficient workers ({}/{})", healthy, required);
                    }
                    _ => {}
                }
                let health_map = fd_for_events.all_worker_health();
                let mut ft = ft_status_for_events.write();
                ft.healthy_workers = health_map.values().filter(|w| w.health == WorkerHealth::Healthy).count();
                ft.degraded_workers = health_map.values().filter(|w| w.health == WorkerHealth::Degraded).count();
                ft.failed_workers = health_map.values().filter(|w| w.health == WorkerHealth::Failed).count();
                ft.system_healthy = fd_for_events.is_system_healthy();
            }
        });

        // --- 4. RPC Server ---
        self.health.write().rpc_server = SubsystemStatus::Starting;
        let (round_trigger_tx, mut round_trigger_rx) = broadcast::channel::<()>(16);
        let api_snapshot = Arc::new(RwLock::new(OrchestratorSnapshot::default()));
        let proof_status = Arc::new(RwLock::new(Vec::<ProofStatusEntry>::new()));
        let round_weights_shared = Arc::new(RwLock::new(Vec::new()));

        // --- Model Store (persistent model weight storage) ---
        let model_store = {
            use crate::storage::model_store::{ModelStore, ModelStoreConfig};
            let data_dir = if self.config.checkpoint_dir.as_os_str().is_empty() {
                std::path::PathBuf::from("data")
            } else {
                self.config.checkpoint_dir.join("model_store")
            };
            match ModelStore::new(ModelStoreConfig {
                data_dir,
                ..ModelStoreConfig::default()
            }) {
                Ok(store) => Some(Arc::new(parking_lot::Mutex::new(store))),
                Err(e) => {
                    warn!("Model store unavailable: {}. Weights will not be persisted to disk.", e);
                    None
                }
            }
        };

        // Restore persisted round weights into the in-memory store for RPC access
        if let Some(ref store) = model_store {
            let store_guard = store.lock();
            let entries = store_guard.get_all_manifest_entries();
            let mut restored = 0usize;
            for manifest_entry in &entries {
                // Load weight bytes from disk and populate round_weights
                match store_guard.load_model(manifest_entry.model_id, manifest_entry.round_id) {
                    Ok(weight_bytes) => {
                        round_weights_shared.write().push(
                            crate::api::rpc::RoundWeightEntry {
                                round_id: manifest_entry.round_id,
                                model_id: manifest_entry.model_id,
                                commitment: manifest_entry.commitment,
                                weight_bytes,
                                loss: manifest_entry.loss,
                                error_bound: manifest_entry.error_bound,
                                steps_completed: manifest_entry.steps_completed,
                                num_contributors: manifest_entry.num_contributors,
                                completed_at: manifest_entry.completed_at,
                                tx_hash: manifest_entry.tx_hash.clone(),
                            },
                        );
                        restored += 1;
                    }
                    Err(e) => {
                        warn!(
                            "Failed to restore weights for model_{}/round_{}: {}",
                            manifest_entry.model_id, manifest_entry.round_id, e,
                        );
                    }
                }
            }
            if restored > 0 {
                info!("Restored {} round weight entries from persistent store", restored);
            }
        }

        let rpc_state = self.build_rpc_state(
            api_snapshot.clone(),
            round_trigger_tx.clone(),
            proof_status.clone(),
            round_weights_shared.clone(),
            model_store.clone(),
        );

        let rpc_addr: SocketAddr = format!("0.0.0.0:{}", self.config.rpc_port).parse()?;
        let rpc_state_clone = rpc_state.clone();
        let rpc_handle = tokio::spawn(async move {
            if let Err(e) = start_rpc_server(rpc_addr, rpc_state_clone).await {
                error!("JSON-RPC server error: {}", e);
            }
        });
        self.health.write().rpc_server = SubsystemStatus::Running;
        info!("JSON-RPC server listening on {}", rpc_addr);

        // --- 5. HTTP API ---
        self.health.write().http_api = SubsystemStatus::Starting;
        let api_key = crate::config::ApiConfig::default().api_key_or_generate();
        info!("HTTP API key: {}", api_key);
        let api_state = Arc::new(ApiState {
            orchestrator_workers: api_snapshot.clone(),
            round_trigger_tx: round_trigger_tx.clone(),
            peers: Arc::new(RwLock::new(PeerSnapshot::default())),
            metrics: Arc::new(RwLock::new(MetricsSnapshot::default())),
            api_key,
            rate_limiter: Arc::new(ApiRateLimiter::new(100)),
            round_weights: round_weights_shared.clone(),
            node_role: "aggregator".to_string(),
            start_time: Instant::now(),
            last_checkpoint_ts: Arc::new(RwLock::new(None)),
            mpc_health: Arc::new(RwLock::new(MpcHealthStatus::default())),
            fault_tolerance_status: fault_tolerance_status.clone(),
            node_metrics: self.node_metrics.clone(),
            active_tasks: Arc::new(RwLock::new(Vec::new())),
        });

        let http_addr: SocketAddr = format!("0.0.0.0:{}", self.config.http_port).parse()?;
        let api_state_clone = api_state.clone();
        let http_handle = tokio::spawn(async move {
            if let Err(e) = crate::api::http::start_api_server(http_addr, api_state_clone).await {
                error!("HTTP API error: {}", e);
            }
        });
        self.health.write().http_api = SubsystemStatus::Running;
        info!("HTTP API listening on {}", http_addr);

        // --- 6. MPC Session Orchestrator ---
        let mpc_enabled = self.config.mpc.enabled;
        let mpc_num_parties = self.config.mpc.num_parties;
        let _mpc_session_orch = if mpc_enabled {
            let ckpt_dir = if self.config.checkpoint_dir.as_os_str().is_empty() {
                None
            } else {
                Some(self.config.checkpoint_dir.clone())
            };
            Some(MpcSessionOrchestrator::new(
                t.min_workers.max(mpc_num_parties), ckpt_dir,
            ))
        } else {
            None
        };

        let _private_aggregator: Option<PrivateAggregator> = if mpc_enabled {
            let mpc_config = MPCTrainingConfig {
                num_parties: mpc_num_parties,
                learning_rate: t.learning_rate,
                d_in: t.d_in,
                d_hid: t.d_hid,
                d_out: t.d_out,
                max_gradient_norm: self.config.mpc.max_gradient_norm,
                ..Default::default()
            };
            let model = MlpModel::new_random(t.d_in, t.d_hid, t.d_out, t.model_seed);
            match PrivateAggregator::new(mpc_config, &model) {
                Ok(agg) => Some(agg),
                Err(e) => {
                    warn!("Failed to create MPC aggregator: {}. Using cleartext.", e);
                    None
                }
            }
        } else {
            None
        };

        // --- Persistence ---
        let persistence = self.create_persistence("aggregator");

        // Initialize model
        let initial_model = self.restore_or_create_model(&persistence);
        let model_hash = initial_model.commitment();
        let aggregator_model = Arc::new(RwLock::new(initial_model.clone()));

        let initial_ckpt = initial_model.to_checkpoint(0);
        let initial_ckpt_bytes = initial_ckpt.to_bytes()
            .expect("Failed to serialize initial model checkpoint");
        let current_checkpoint_bytes = Arc::new(RwLock::new(initial_ckpt_bytes));

        info!(
            "Model initialized: {}x{}x{} ({} params), commitment={}",
            t.d_in, t.d_hid, t.d_out,
            initial_model.num_params(),
            hex::encode(&model_hash[..8]),
        );

        // Spawn orchestrator event handler
        let snapshot_for_events = api_snapshot.clone();
        let proof_status_events = proof_status.clone();
        let fd_for_orch = failure_detector.clone();
        let (completed_round_tx, mut completed_round_rx) = tokio::sync::mpsc::channel::<u64>(16);
        let mut completed_rounds = 0u64;
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                match &event {
                    OrchestratorEvent::WorkerJoined { peer_id } => {
                        info!("Worker joined: {}", peer_id);
                        fd_for_orch.register_worker(peer_id.clone());
                    }
                    OrchestratorEvent::WorkerLeft { peer_id } => {
                        info!("Worker left: {}", peer_id);
                        fd_for_orch.unregister_worker(peer_id);
                    }
                    OrchestratorEvent::RoundStarted { round_id, workers } => {
                        info!("Round {} started with {} workers", round_id, workers.len());
                        proof_status_events.write().push(ProofStatusEntry {
                            round_id: *round_id,
                            status: "collecting".to_string(),
                            proofs_collected: 0,
                        });
                    }
                    OrchestratorEvent::GradientReceived { round_id, peer_id } => {
                        info!("Gradient for round {} from {}", round_id, peer_id);
                        let mut status = proof_status_events.write();
                        if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                            entry.proofs_collected += 1;
                        }
                    }
                    OrchestratorEvent::RoundCompleted { round_id, result_hash } => {
                        completed_rounds += 1;
                        snapshot_for_events.write().completed_rounds = completed_rounds;
                        {
                            let mut status = proof_status_events.write();
                            if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                                entry.status = "completed".to_string();
                            }
                        }
                        info!("Round {} completed, result={}", round_id, hex::encode(&result_hash[..8]));
                        let _ = completed_round_tx.send(*round_id).await;
                    }
                    OrchestratorEvent::RoundFailed { round_id, reason } => {
                        error!("Round {} failed: {}", round_id, reason);
                        let mut status = proof_status_events.write();
                        if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                            entry.status = "failed".to_string();
                        }
                    }
                    _ => {}
                }
            }
        });

        // --- Main aggregator loop (TrainingJobManager-driven) ---
        //
        // When helix_startTraining fires on round_trigger_rx, the aggregator creates
        // a TrainingJobManager and runs it to completion. The job manager handles:
        //   1. Announcing the round via RoundManagementMessage::RoundConfigure
        //   2. Collecting worker readiness (quorum)
        //   3. Distributing model weights
        //   4. Collecting proofs/gradients from workers
        //   5. FedAvg aggregation with Byzantine filtering
        //   6. On-chain proof submission (if chain config present)
        //   7. Broadcasting results and updated weights
        //
        // Network events are forwarded to the job manager via a message channel.
        // The orchestrator's legacy event handler is kept for API/dashboard updates.

        use crate::training::job_manager::{
            TrainingJobManager, TrainingJobConfig, WorkerMessage,
            network_event_to_worker_message,
        };
        use crate::training::fault_recovery::{
            GracefulShutdownCoordinator, WorkerFailureHandler, FaultRecoveryConfig,
        };

        // --- Fault Recovery ---
        let fault_recovery_config = FaultRecoveryConfig::from_min_workers(t.min_workers);
        let shutdown_coordinator = Arc::new(GracefulShutdownCoordinator::new());
        let _worker_failure_handler = WorkerFailureHandler::new(fault_recovery_config.min_workers);
        let _shutdown_handle = shutdown_coordinator.install_signal_handlers();
        info!("Graceful shutdown coordinator installed");

        let mut round_number = 0u64;
        let mut total_rounds_completed = 0u64;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // --- Chain Watcher ---
        // Resolve the signer address for local slash detection
        let signer_address = self.config.private_key().and_then(|pk| {
            use ethers::signers::LocalWallet;
            use std::str::FromStr;
            LocalWallet::from_str(&pk).ok().map(|w| {
                use ethers::signers::Signer;
                w.address()
            })
        });
        let _chain_watcher = self.start_chain_watcher(signer_address).await;

        // On-chain pipeline reference (if configured)
        let on_chain_pipeline: Option<Arc<crate::on_chain_pipeline::OnChainPipeline>> =
            if let Some(ref chain_config) = self.config.chain {
                if let Some(ref pk) = self.config.private_key() {
                    match crate::on_chain_pipeline::OnChainPipeline::new(
                        &self.config.rpc_url, pk, chain_config.clone(),
                    ).await {
                        Ok(pipeline) => {
                            let pipeline = Arc::new(pipeline);
                            // Initialize: register model + stake
                            match pipeline.initialize().await {
                                Ok(model_id) => {
                                    info!("On-chain pipeline initialized: model_id={}", model_id);
                                    self.health.write().chain_pipeline = SubsystemStatus::Running;
                                }
                                Err(e) => {
                                    warn!("On-chain initialization failed: {}. Continuing offline.", e);
                                }
                            }
                            Some(pipeline)
                        }
                        Err(e) => {
                            warn!("Failed to create on-chain pipeline: {}. Continuing offline.", e);
                            None
                        }
                    }
                } else {
                    warn!("Chain config present but no private key set (HELIX_PRIVATE_KEY). Skipping on-chain.");
                    None
                }
            } else {
                None
            };

        tokio::select! {
            _ = shutdown_rx.changed() => {
                info!("Aggregator shutting down gracefully...");

                // Notify workers
                network.broadcast(MessagePayload::Training(
                    TrainingMessage::RoundComplete {
                        round_id: round_number,
                        result_hash: aggregator_model.read().commitment(),
                    },
                )).await;

                // Stop failure detector
                fd_handle.abort();

                // Save aggregator state
                if let Some(ref p) = persistence {
                    let model_ckpt = aggregator_model.read().to_checkpoint(round_number);
                    if let Ok(ckpt_bytes) = model_ckpt.to_bytes() {
                        let snapshot = AggregatorSnapshot::new(
                            ckpt_bytes,
                            round_number,
                            total_rounds_completed,
                            t.d_in, t.d_hid, t.d_out,
                            t.model_seed,
                            t.learning_rate,
                        );
                        if let Err(e) = p.save_aggregator(&snapshot) {
                            error!("Failed to save aggregator checkpoint: {}", e);
                        } else {
                            info!("Aggregator checkpoint saved (round={})", round_number);
                        }
                    }
                }
            }
            _ = async {
                loop {
                    // Also handle legacy orchestrator completed rounds
                    while let Ok(completed_round_id) = completed_round_rx.try_recv() {
                        let updates = orchestrator.take_weight_updates(completed_round_id);
                        if updates.is_empty() {
                            continue;
                        }
                        let mut worker_models = Vec::new();
                        for (_peer_id, ckpt_bytes, _hash) in &updates {
                            if let Ok(ckpt) = ModelCheckpoint::from_bytes(ckpt_bytes) {
                                if let Ok(model) = MlpModel::from_checkpoint(&ckpt) {
                                    worker_models.push(model);
                                }
                            }
                        }
                        if !worker_models.is_empty() {
                            if let Ok(aggregated) = average_models(&worker_models) {
                                let new_hash = aggregated.commitment();
                                *aggregator_model.write() = aggregated.clone();
                                total_rounds_completed += 1;
                                let ckpt = aggregated.to_checkpoint(completed_round_id);
                                if let Ok(bytes) = ckpt.to_bytes() {
                                    *current_checkpoint_bytes.write() = bytes.clone();

                                    // Publish to RPC round_weights (legacy orchestrator path)
                                    {
                                        use sha2::{Sha256, Digest};
                                        let mut hasher = Sha256::new();
                                        hasher.update(&bytes);
                                        let hash = hasher.finalize();
                                        let mut commitment = [0u8; 32];
                                        commitment.copy_from_slice(&hash);
                                        let now_ts = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .map(|d| d.as_secs())
                                            .unwrap_or(0);
                                        round_weights_shared.write().push(
                                            crate::api::rpc::RoundWeightEntry {
                                                round_id: completed_round_id,
                                                model_id: 0,
                                                commitment,
                                                weight_bytes: bytes.clone(),
                                                loss: 0.0,
                                                error_bound: 0.0,
                                                steps_completed: completed_round_id,
                                                num_contributors: worker_models.len() as u32,
                                                completed_at: now_ts,
                                                tx_hash: None,
                                            },
                                        );
                                    }

                                    // Persist to model store (legacy path)
                                    if let Some(ref store) = model_store {
                                        use crate::storage::model_store::ModelStoreMetadata;
                                        let metadata = ModelStoreMetadata {
                                            num_contributors: worker_models.len() as u32,
                                            loss: 0.0,
                                            error_bound: 0.0,
                                            steps_completed: completed_round_id,
                                            tx_hash: None,
                                        };
                                        if let Err(e) = store.lock().store_model(0, completed_round_id, &bytes, metadata) {
                                            error!("Failed to persist model (legacy): {}", e);
                                        }
                                    }

                                    network.broadcast(MessagePayload::Training(
                                        TrainingMessage::UpdatedWeights {
                                            round_id: completed_round_id,
                                            checkpoint_data: bytes,
                                            weight_hash: new_hash,
                                        },
                                    )).await;
                                }
                            }
                        }
                    }

                    // Wait for new round trigger
                    tokio::select! {
                        _ = round_trigger_rx.recv() => {
                            round_number += 1;

                            // Check graceful shutdown before starting new round
                            if !shutdown_coordinator.should_accept_new_round() {
                                info!("Shutdown requested, declining new round {}", round_number);
                                round_number -= 1;
                                continue;
                            }

                            shutdown_coordinator.register_active_round(round_number);
                            info!("=== Round {} triggered via RPC ===", round_number);

                            // Build job config from node config
                            let job_config = TrainingJobConfig::from_node_config(&self.config);

                            // Create a message channel for forwarding network events
                            let (worker_msg_tx, worker_msg_rx) =
                                tokio::sync::mpsc::channel::<WorkerMessage>(1000);

                            // Create the job manager with current model state
                            let current_model = aggregator_model.read().clone();
                            let mut job_manager = TrainingJobManager::new(
                                local_id.clone(),
                                job_config,
                                network.clone(),
                                on_chain_pipeline.clone(),
                                current_model,
                            );

                            // Subscribe to job events for API/dashboard updates
                            let mut job_event_rx = job_manager.subscribe_events();
                            let snapshot_for_job = api_snapshot.clone();
                            let proof_status_for_job = proof_status.clone();
                            tokio::spawn(async move {
                                while let Ok(event) = job_event_rx.recv().await {
                                    match &event {
                                        crate::training::job_manager::JobEvent::ProofReceived {
                                            round_id, proofs_received, ..
                                        } => {
                                            let mut status = proof_status_for_job.write();
                                            if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                                                entry.proofs_collected = *proofs_received;
                                            } else {
                                                status.push(ProofStatusEntry {
                                                    round_id: *round_id,
                                                    status: "collecting".to_string(),
                                                    proofs_collected: *proofs_received,
                                                });
                                            }
                                        }
                                        crate::training::job_manager::JobEvent::RoundComplete {
                                            round_id, num_contributors, ..
                                        } => {
                                            snapshot_for_job.write().completed_rounds += 1;
                                            let mut status = proof_status_for_job.write();
                                            if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                                                entry.status = "completed".to_string();
                                                entry.proofs_collected = *num_contributors;
                                            }
                                        }
                                        crate::training::job_manager::JobEvent::RoundFailed {
                                            round_id, reason, ..
                                        } => {
                                            let mut status = proof_status_for_job.write();
                                            if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                                                entry.status = format!("failed: {}", reason);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            });

                            // Spawn a task to forward network events to the job manager
                            let net_for_forward = network.clone();
                            let worker_msg_tx_clone = worker_msg_tx.clone();
                            let forward_handle = tokio::spawn(async move {
                                let event_rx = net_for_forward.event_receiver();
                                let mut rx = event_rx.lock().await;
                                while let Some(event) = rx.recv().await {
                                    let (from, payload) = match &event {
                                        crate::network::runner::NetworkEvent::RoundManagementMessage { from, message } => {
                                            (from.clone(), MessagePayload::RoundManagement(message.clone()))
                                        }
                                        crate::network::runner::NetworkEvent::TrainingMessage { from, message } => {
                                            (from.clone(), MessagePayload::Training(message.clone()))
                                        }
                                        crate::network::runner::NetworkEvent::GradientMessage { from, message } => {
                                            (from.clone(), MessagePayload::Gradient(message.clone()))
                                        }
                                        crate::network::runner::NetworkEvent::RegistrationMessage { from, message } => {
                                            (from.clone(), MessagePayload::Registration(message.clone()))
                                        }
                                        _ => continue,
                                    };

                                    if let Some(worker_msg) = network_event_to_worker_message(from, &payload) {
                                        if worker_msg_tx_clone.send(worker_msg).await.is_err() {
                                            break; // Job manager dropped its receiver
                                        }
                                    }
                                }
                            });

                            // Run the job manager
                            match job_manager.run(worker_msg_rx).await {
                                Ok(result) => {
                                    // Update shared model state
                                    *aggregator_model.write() = result.aggregated_model.clone();
                                    total_rounds_completed += 1;
                                    let ckpt = result.aggregated_model.to_checkpoint(round_number);
                                    let weight_bytes = ckpt.to_bytes().unwrap_or_default();
                                    *current_checkpoint_bytes.write() = weight_bytes.clone();

                                    // Compute SHA-256 commitment of weight bytes
                                    let commitment = {
                                        use sha2::{Sha256, Digest};
                                        let mut hasher = Sha256::new();
                                        hasher.update(&weight_bytes);
                                        let hash = hasher.finalize();
                                        let mut out = [0u8; 32];
                                        out.copy_from_slice(&hash);
                                        out
                                    };

                                    // Determine model ID from on-chain pipeline or default
                                    let model_id = if let Some(ref pipeline) = on_chain_pipeline {
                                        pipeline.state().await.model_id.unwrap_or(0)
                                    } else {
                                        0
                                    };

                                    let now_ts = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|d| d.as_secs())
                                        .unwrap_or(0);

                                    // Publish result to RPC round_weights for retrieval
                                    {
                                        let entry = crate::api::rpc::RoundWeightEntry {
                                            round_id: round_number,
                                            model_id,
                                            commitment,
                                            weight_bytes,
                                            loss: result.avg_loss,
                                            error_bound: result.total_error_bound,
                                            steps_completed: result.steps_completed,
                                            num_contributors: result.num_contributors as u32,
                                            completed_at: now_ts,
                                            tx_hash: result.tx_hash.clone(),
                                        };
                                        round_weights_shared.write().push(entry);
                                    }

                                    // Persist to model store for durable retrieval
                                    if let Some(ref store) = model_store {
                                        use crate::storage::model_store::ModelStoreMetadata;
                                        let metadata = ModelStoreMetadata {
                                            num_contributors: result.num_contributors as u32,
                                            loss: result.avg_loss,
                                            error_bound: result.total_error_bound,
                                            steps_completed: result.steps_completed,
                                            tx_hash: result.tx_hash.clone(),
                                        };
                                        let wb = current_checkpoint_bytes.read().clone();
                                        match store.lock().store_model(model_id, round_number, &wb, metadata) {
                                            Ok(entry) => {
                                                info!(
                                                    "Model persisted: model_id={}, round={}, {} bytes, commitment={}",
                                                    model_id, round_number, entry.size_bytes,
                                                    hex::encode(entry.commitment),
                                                );
                                            }
                                            Err(e) => {
                                                error!("Failed to persist model weights: {}", e);
                                            }
                                        }
                                    }

                                    info!(
                                        "TrainingJobManager round {} complete: {} contributors, loss={:.6}, error={:.6}, tx={:?}",
                                        round_number,
                                        result.num_contributors,
                                        result.avg_loss,
                                        result.total_error_bound,
                                        result.tx_hash,
                                    );

                                    // Finalize round on-chain (triggers reward distribution)
                                    if let Some(ref pipeline) = on_chain_pipeline {
                                        match pipeline.finalize_round(round_number).await {
                                            Ok(tx) => info!("Round {} finalized on-chain: tx={}", round_number, tx),
                                            Err(e) => {
                                                // May fail if auto-finalized by submitProof or dispute pending
                                                warn!("Round {} on-chain finalization skipped: {}", round_number, e);
                                            }
                                        }
                                    }

                                    // Save checkpoint
                                    if let Some(ref p) = persistence {
                                        let model_ckpt = result.aggregated_model.to_checkpoint(round_number);
                                        if let Ok(ckpt_bytes) = model_ckpt.to_bytes() {
                                            let snapshot = AggregatorSnapshot::new(
                                                ckpt_bytes,
                                                round_number,
                                                total_rounds_completed,
                                                t.d_in, t.d_hid, t.d_out,
                                                t.model_seed,
                                                t.learning_rate,
                                            );
                                            if let Err(e) = p.save_aggregator(&snapshot) {
                                                error!("Failed to save checkpoint: {}", e);
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("TrainingJobManager round {} failed: {}", round_number, e);
                                }
                            }

                            // Clear active round for shutdown coordinator
                            shutdown_coordinator.clear_active_round();

                            // Clean up the forwarding task
                            forward_handle.abort();
                        }
                        Some(rid) = completed_round_rx.recv() => {
                            // Handle legacy orchestrator completed rounds
                            let updates = orchestrator.take_weight_updates(rid);
                            if !updates.is_empty() {
                                let mut worker_models = Vec::new();
                                for (_, ckpt_bytes, _) in &updates {
                                    if let Ok(ckpt) = ModelCheckpoint::from_bytes(ckpt_bytes) {
                                        if let Ok(model) = MlpModel::from_checkpoint(&ckpt) {
                                            worker_models.push(model);
                                        }
                                    }
                                }
                                if !worker_models.is_empty() {
                                    if let Ok(aggregated) = average_models(&worker_models) {
                                        *aggregator_model.write() = aggregated;
                                        total_rounds_completed += 1;
                                    }
                                }
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(1)) => {
                            // Periodic tick for housekeeping
                        }
                    }
                }
            } => {}
        }

        // Cleanup
        rpc_handle.abort();
        http_handle.abort();
        self.health.write().network = SubsystemStatus::Stopped;
        self.health.write().rpc_server = SubsystemStatus::Stopped;
        self.health.write().http_api = SubsystemStatus::Stopped;
        self.health.write().training_orchestrator = SubsystemStatus::Stopped;
        self.health.write().failure_detector = SubsystemStatus::Stopped;
        info!("Aggregator shutdown complete (rounds={})", total_rounds_completed);
        Ok(())
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn create_persistence(&self, prefix: &str) -> Option<StatePersistence> {
        let dir = &self.config.checkpoint_dir;
        if dir.as_os_str().is_empty() {
            return None;
        }
        match StatePersistence::new(
            Some(dir.clone()),
            self.config.fault_tolerance.max_checkpoints,
            prefix,
        ) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("Failed to initialize persistence: {}", e);
                None
            }
        }
    }

    fn restore_or_create_model(&self, persistence: &Option<StatePersistence>) -> MlpModel {
        let t = &self.config.training;

        if let Some(ref p) = persistence {
            if let Ok(Some(snap)) = p.load_latest_aggregator() {
                if let Ok(ckpt) = ModelCheckpoint::from_bytes(&snap.model_checkpoint_bytes) {
                    if let Ok(model) = MlpModel::from_checkpoint(&ckpt) {
                        info!(
                            "Restored model from checkpoint (round={}, step={})",
                            snap.last_completed_round, ckpt.step_number,
                        );
                        return model;
                    }
                }
            }
        }

        MlpModel::new_random(t.d_in, t.d_hid, t.d_out, t.model_seed)
    }
}

// ============================================================================
// Signal Handler
// ============================================================================

/// Creates a future that resolves on SIGINT or SIGTERM.
pub async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => info!("Received SIGINT, initiating graceful shutdown..."),
            _ = sigterm.recv() => info!("Received SIGTERM, initiating graceful shutdown..."),
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
        info!("Received SIGINT, initiating graceful shutdown...");
    }
}

// ============================================================================
// Training Data Generation (kept for compatibility)
// ============================================================================

/// Generates synthetic training data for a given seed.
pub fn generate_training_data(d_in: usize, d_out: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x: Vec<f64> = (0..d_in).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f64> = (0..d_out).map(|_| rng.gen_range(0.0..1.0)).collect();
    (x, target)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subsystem_status_display() {
        assert_eq!(SubsystemStatus::Pending.to_string(), "pending");
        assert_eq!(SubsystemStatus::Running.to_string(), "running");
        assert_eq!(
            SubsystemStatus::Failed("oops".into()).to_string(),
            "failed: oops"
        );
    }

    #[test]
    fn test_runtime_health_initial_state() {
        let health = RuntimeHealth::new();
        assert!(!health.is_healthy());
        assert_eq!(health.network, SubsystemStatus::Pending);
    }

    #[test]
    fn test_runtime_health_healthy_when_core_running() {
        let mut health = RuntimeHealth::new();
        health.network = SubsystemStatus::Running;
        health.rpc_server = SubsystemStatus::Running;
        assert!(health.is_healthy());
    }

    #[test]
    fn test_runtime_health_unhealthy_when_network_failed() {
        let mut health = RuntimeHealth::new();
        health.network = SubsystemStatus::Failed("test".into());
        health.rpc_server = SubsystemStatus::Running;
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_runtime_health_json() {
        let mut health = RuntimeHealth::new();
        health.network = SubsystemStatus::Running;
        health.rpc_server = SubsystemStatus::Running;
        let json = health.to_json();
        assert_eq!(json["healthy"], true);
        assert!(json["uptime_secs"].is_number());
        assert!(json["components"]["network"].is_string());
    }

    #[test]
    fn test_generate_training_data() {
        let (x, t) = generate_training_data(4, 2, 42);
        assert_eq!(x.len(), 4);
        assert_eq!(t.len(), 2);

        // Same seed produces same data
        let (x2, t2) = generate_training_data(4, 2, 42);
        assert_eq!(x, x2);
        assert_eq!(t, t2);

        // Different seed produces different data
        let (x3, _) = generate_training_data(4, 2, 43);
        assert_ne!(x, x3);
    }

    #[test]
    fn test_runtime_new() {
        let config = NodeConfig::default();
        let runtime = NodeRuntime::new(config);
        assert!(!runtime.health().is_healthy());
    }

    #[test]
    fn test_runtime_shutdown() {
        let config = NodeConfig::default();
        let runtime = NodeRuntime::new(config);
        runtime.shutdown();
        // Should not panic
    }
}
