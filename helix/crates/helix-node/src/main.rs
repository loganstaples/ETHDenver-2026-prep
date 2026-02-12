mod sc_client;

use helix_node::sc_client::SCClient;
use helix_node::api::http::{ApiRateLimiter, ApiState, MetricsSnapshot, OrchestratorSnapshot, PeerSnapshot, RoundInfo};
use helix_node::api::rpc::{
    ProofStatusEntry, RpcRateLimiter, RpcState, start_rpc_server,
    NodeConfigSnapshot, MPCStatusSnapshot,
};
use helix_node::network::messages::{
    GradientMessage, HeartbeatMessage, MessagePayload, NodeCapabilities, PeerId, PeerInfo,
    TrainingMessage, TrainingParams,
};
use helix_node::network::runner::{NetworkEvent, NetworkRunnerBuilder};
use helix_node::round_commit::{RoundCommitConfig, RoundCommitManager};
use helix_node::trainer::Trainer;
use helix_node::training::orchestrator::{
    OrchestratorConfig, OrchestratorEvent, TrainingOrchestrator,
};
use helix_node::training::MPCWorkerHandle;
use helix_node::trainer::MlpModel;

use log::{error, info, warn};
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Reads an env var or returns a default.
fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Creates a shutdown signal future that resolves on SIGINT or SIGTERM.
async fn shutdown_signal() {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    let role = env_or("HELIX_NODE_ROLE", "worker");
    let listen_addr: SocketAddr = env_or("HELIX_LISTEN_ADDR", "127.0.0.1:9000")
        .parse()
        .expect("invalid HELIX_LISTEN_ADDR");

    info!("Starting Helix Node (role={}, addr={})", role, listen_addr);

    match role.as_str() {
        "worker" => run_worker(listen_addr).await,
        "aggregator" => run_aggregator(listen_addr).await,
        _ => {
            error!("Unknown HELIX_NODE_ROLE: '{}'. Use 'worker' or 'aggregator'.", role);
            std::process::exit(1);
        }
    }
}

// ============================================================================
// Worker Role
// ============================================================================

async fn run_worker(listen_addr: SocketAddr) -> anyhow::Result<()> {
    let aggregator_addr: SocketAddr = env_or("HELIX_AGGREGATOR_ADDR", "127.0.0.1:9000")
        .parse()
        .expect("invalid HELIX_AGGREGATOR_ADDR");
    let rpc_port: u16 = env_or("HELIX_RPC_PORT", "9002").parse().unwrap_or(9002);

    // MPC configuration
    let mpc_enabled = env_or("HELIX_MPC_ENABLED", "0") == "1";
    let mpc_party_index: usize = env_or("HELIX_MPC_PARTY_INDEX", "0").parse().unwrap_or(0);
    let mpc_num_parties: usize = env_or("HELIX_MPC_NUM_PARTIES", "3").parse().unwrap_or(3);

    if mpc_enabled {
        info!(
            "MPC training enabled: party {}/{} (index/total)",
            mpc_party_index, mpc_num_parties,
        );
    }

    let local_id = PeerId::random();
    info!("Worker {} starting, will connect to aggregator at {}", local_id, aggregator_addr);

    // Build network runner
    let network = NetworkRunnerBuilder::new()
        .local_id(local_id.clone())
        .listen_addr(listen_addr)
        .capabilities(NodeCapabilities {
            can_train: true,
            can_aggregate: false,
            can_prove: true,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 10,
        })
        .build()?;

    let network = Arc::new(network);
    network.start().await?;

    // Connect to aggregator
    let agg_peer = PeerInfo {
        id: PeerId::from_string("aggregator"),
        address: aggregator_addr.to_string(),
        capabilities: NodeCapabilities {
            can_aggregate: true,
            ..Default::default()
        },
        last_seen: 0,
        reputation: 100,
    };
    network.connect_peer(agg_peer).await?;
    info!("Connected to aggregator at {}", aggregator_addr);

    // Set up shared state for RPC
    let (round_trigger_tx, _) = broadcast::channel::<()>(16);
    let api_snapshot = Arc::new(RwLock::new(OrchestratorSnapshot::default()));
    let proof_status = Arc::new(RwLock::new(Vec::<ProofStatusEntry>::new()));

    let (stop_trigger_tx_w, _) = broadcast::channel::<()>(16);
    let rpc_state = Arc::new(RpcState {
        snapshot: api_snapshot.clone(),
        round_trigger_tx: round_trigger_tx.clone(),
        stop_trigger_tx: stop_trigger_tx_w,
        node_role: "worker".to_string(),
        start_time: std::time::Instant::now(),
        proof_status: proof_status.clone(),
        proof_queue: Arc::new(RwLock::new(Vec::new())),
        node_config: Arc::new(RwLock::new(NodeConfigSnapshot::default())),
        aggregation_results: Arc::new(RwLock::new(Vec::new())),
        mpc_status: Arc::new(RwLock::new(MPCStatusSnapshot::default())),
        model_id: Arc::new(RwLock::new(None)),
        rpc_addr: format!("0.0.0.0:{}", rpc_port),
        rate_limiter: Arc::new(RwLock::new(RpcRateLimiter::default())),
    });

    // Start JSON-RPC server
    let rpc_addr: SocketAddr = format!("0.0.0.0:{}", rpc_port).parse().unwrap();
    let rpc_state_clone = rpc_state.clone();
    tokio::spawn(async move {
        if let Err(e) = start_rpc_server(rpc_addr, rpc_state_clone).await {
            error!("JSON-RPC server error: {}", e);
        }
    });

    // Heartbeat loop: send heartbeats to aggregator so it registers us
    let network_hb = network.clone();
    let hb_handle = tokio::spawn(async move {
        let mut seq = 0u64;
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            network_hb
                .broadcast(MessagePayload::Heartbeat(HeartbeatMessage {
                    seq,
                    is_pong: false,
                    load: 30,
                }))
                .await;
            seq += 1;
        }
    });

    // Main event loop: wait for RoundStart, train, send gradient
    let mut trainer: Option<Trainer> = None;
    let mut mpc_handle: Option<MPCWorkerHandle> = None;
    let mut steps_completed = 0u64;

    let result = tokio::select! {
        _ = shutdown_signal() => {
            info!("Worker shutting down...");
            Ok(())
        }
        result = async {
            loop {
                match network.next_event().await {
                    Some(NetworkEvent::TrainingMessage { from, message }) => {
                        match message {
                            TrainingMessage::RoundStart {
                                round_id,
                                model_hash: _,
                                params,
                            } => {
                                info!(
                                    "Received RoundStart #{} from {} (model {}x{}x{}, lr={}, seed={})",
                                    round_id,
                                    from,
                                    params.d_in,
                                    params.d_hid,
                                    params.d_out,
                                    params.learning_rate,
                                    params.model_seed,
                                );

                                if mpc_enabled {
                                    // ── MPC training path with ZK proof generation ──
                                    if mpc_handle.is_none() {
                                        let model = MlpModel::new_random(
                                            params.d_in,
                                            params.d_hid,
                                            params.d_out,
                                            params.model_seed,
                                        );
                                        mpc_handle = Some(MPCWorkerHandle::new(
                                            mpc_party_index,
                                            mpc_num_parties,
                                            model,
                                            params.learning_rate,
                                            100.0, // max gradient norm
                                        ));
                                        info!(
                                            "MPC worker initialized: party {}/{}",
                                            mpc_party_index, mpc_num_parties,
                                        );
                                    }

                                    // Initialize a Trainer in lockstep for ZK proof generation.
                                    // The Trainer uses the same model state as the MPC handle.
                                    if trainer.is_none() {
                                        trainer = Some(Trainer::new(
                                            params.d_in,
                                            params.d_hid,
                                            params.d_out,
                                            params.learning_rate,
                                            params.model_seed,
                                        ));
                                        info!("MPC proof trainer initialized");
                                    }

                                    let mpc = mpc_handle.as_mut().unwrap();
                                    let (x, target) = generate_training_data(
                                        params.d_in,
                                        params.d_out,
                                        params.model_seed + round_id,
                                    );

                                    info!(
                                        "MPC training step (party {}, round {})...",
                                        mpc_party_index, round_id,
                                    );
                                    match mpc.compute_gradient_share(&x, &target) {
                                        Ok(computation) => {
                                            info!(
                                                "MPC gradient share computed: loss={:.6}, commitment={}",
                                                computation.local_loss,
                                                hex::encode(&computation.gradient_commitment[..4]),
                                            );

                                            // Generate ZK proof using the Trainer (same model, same data).
                                            // This proves the training step was computed correctly.
                                            let proof_bytes = if let Some(ref mut t) = trainer {
                                                match t.train_step(&x, &target) {
                                                    Ok(proved_step) => {
                                                        let evm_len = proved_step.evm_bundle.as_ref().map(|b| b.evm_proof.len()).unwrap_or(0);
                                                        info!(
                                                            "MPC ZK proof generated: {} bytes (EVM: {}), verified={}",
                                                            proved_step.proof_result.proof.len(),
                                                            evm_len,
                                                            proved_step.proof_result.verified,
                                                        );
                                                        proved_step.evm_bundle
                                                            .as_ref()
                                                            .map(|b| b.evm_proof.clone())
                                                            .unwrap_or_else(|| proved_step.proof_result.proof.clone())
                                                    }
                                                    Err(e) => {
                                                        warn!("MPC proof generation failed (non-fatal): {}", e);
                                                        computation.gradient_commitment.to_vec()
                                                    }
                                                }
                                            } else {
                                                computation.gradient_commitment.to_vec()
                                            };

                                            let (hiding_commitment, nonce) = helix_node::training::consensus::compute_hiding_gradient_commitment(&computation.gradient_commitment);
                                            network
                                                .broadcast(MessagePayload::Gradient(
                                                    GradientMessage::ShareGradient {
                                                        round_id,
                                                        gradient_commitment: hiding_commitment,
                                                        commitment_nonce: nonce,
                                                        error_bound: computation.local_loss * 0.01,
                                                        proof: proof_bytes,
                                                    },
                                                ))
                                                .await;

                                            steps_completed += 1;
                                            info!(
                                                "MPC gradient share + proof sent for round {} (total steps: {})",
                                                round_id, steps_completed,
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
                                        Err(e) => {
                                            error!("MPC training failed for round {}: {}", round_id, e);
                                        }
                                    }
                                } else {
                                    // ── Regular (non-MPC) training path ────────────
                                    // Initialize trainer if needed (same seed = same initial model)
                                    if trainer.is_none() {
                                        trainer = Some(Trainer::new(
                                            params.d_in,
                                            params.d_hid,
                                            params.d_out,
                                            params.learning_rate,
                                            params.model_seed,
                                        ));
                                    }

                                    let t = trainer.as_mut().unwrap();

                                    // Generate synthetic data from seed (deterministic)
                                    let (x, target) = generate_training_data(params.d_in, params.d_out, params.model_seed + round_id);

                                    info!("Training step {} (round {})...", t.step_count() + 1, round_id);
                                    match t.train_step(&x, &target) {
                                        Ok(result) => {
                                            let evm_proof_len = result.evm_bundle.as_ref().map(|b| b.evm_proof.len()).unwrap_or(0);
                                            info!(
                                                "Proof generated: {} bytes (EVM: {}), loss={:.6}, verified={}, commitment={:?}",
                                                result.proof_result.proof.len(),
                                                evm_proof_len,
                                                result.loss,
                                                result.proof_result.verified,
                                                hex::encode(&result.commitment[..4]),
                                            );

                                            // Use EVM-formatted proof if available, otherwise raw
                                            let proof_bytes = result.evm_bundle
                                                .as_ref()
                                                .map(|b| b.evm_proof.clone())
                                                .unwrap_or_else(|| result.proof_result.proof.clone());

                                            // Send gradient + proof back to aggregator
                                            let (hiding_commitment, nonce) = helix_node::training::consensus::compute_hiding_gradient_commitment(&result.commitment);
                                            network
                                                .broadcast(MessagePayload::Gradient(
                                                    GradientMessage::ShareGradient {
                                                        round_id,
                                                        gradient_commitment: hiding_commitment,
                                                        commitment_nonce: nonce,
                                                        error_bound: result.loss * 0.01,
                                                        proof: proof_bytes,
                                                    },
                                                ))
                                                .await;

                                            steps_completed += 1;
                                            info!("Gradient sent for round {} (total steps: {})", round_id, steps_completed);

                                            // Update proof status
                                            proof_status.write().push(ProofStatusEntry {
                                                round_id,
                                                status: "submitted".to_string(),
                                                proofs_collected: 1,
                                            });
                                        }
                                        Err(e) => {
                                            error!("Training failed for round {}: {}", round_id, e);
                                        }
                                    }
                                }
                            }
                            TrainingMessage::RoundComplete { round_id, .. } => {
                                info!("Round {} completed", round_id);
                            }
                            _ => {}
                        }
                    }
                    Some(NetworkEvent::Heartbeat { .. }) => {
                        // Heartbeat pongs handled automatically
                    }
                    Some(NetworkEvent::PeerDiscovered(peer)) => {
                        info!("Peer discovered: {} at {}", peer.id, peer.address);
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
            Ok::<(), anyhow::Error>(())
        } => result,
    };

    // Graceful cleanup
    hb_handle.abort();
    info!("Worker shutdown complete (steps_completed={})", steps_completed);
    result
}

// ============================================================================
// Aggregator Role
// ============================================================================

async fn run_aggregator(listen_addr: SocketAddr) -> anyhow::Result<()> {
    let min_workers: usize = env_or("HELIX_MIN_WORKERS", "1").parse().unwrap_or(1);
    let d_in: usize = env_or("HELIX_D_IN", "4").parse().unwrap_or(4);
    let d_hid: usize = env_or("HELIX_D_HID", "8").parse().unwrap_or(8);
    let d_out: usize = env_or("HELIX_D_OUT", "2").parse().unwrap_or(2);
    let model_seed: u64 = env_or("HELIX_MODEL_SEED", "42").parse().unwrap_or(42);
    let learning_rate: f64 = env_or("HELIX_LR", "0.01").parse().unwrap_or(0.01);
    let http_port: u16 = env_or("HELIX_HTTP_PORT", "9001").parse().unwrap_or(9001);
    let rpc_port: u16 = env_or("HELIX_RPC_PORT", "9002").parse().unwrap_or(9002);

    let local_id = PeerId::from_string("aggregator");
    info!(
        "Aggregator starting on {} (min_workers={}, model={}x{}x{})",
        listen_addr, min_workers, d_in, d_hid, d_out
    );

    // Build network runner
    let network = NetworkRunnerBuilder::new()
        .local_id(local_id.clone())
        .listen_addr(listen_addr)
        .capabilities(NodeCapabilities {
            can_train: false,
            can_aggregate: true,
            can_prove: false,
            gpu_memory_mb: 0,
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            storage_gb: 10,
        })
        .build()?;

    let network = Arc::new(network);
    network.start().await?;
    info!("Aggregator listening on {}", listen_addr);

    // Create orchestrator
    let orch_config = OrchestratorConfig {
        min_workers,
        collection_timeout: Duration::from_secs(120),
        heartbeat_interval: Duration::from_secs(10),
        worker_timeout: Duration::from_secs(60),
        default_params: TrainingParams {
            learning_rate,
            batch_size: 1,
            local_epochs: 1,
            max_error_bound: 0.1,
            d_in,
            d_hid,
            d_out,
            model_seed,
        },
        ..Default::default()
    };

    let orchestrator = TrainingOrchestrator::new(local_id.clone(), orch_config, network.clone());
    orchestrator.set_leader(true);

    // Wire up RoundCommitManager with SCClient if env vars are set
    let rpc_url = env_or("HELIX_ETH_RPC", "");
    let private_key = env_or("PRIVATE_KEY", "");
    let coordinator_addr = env_or("COORDINATOR_ADDRESS", "");

    if !rpc_url.is_empty() && !private_key.is_empty() && !coordinator_addr.is_empty() {
        info!("Connecting to chain at {} (coordinator={})", rpc_url, coordinator_addr);
        match SCClient::with_config(&rpc_url, &private_key, &coordinator_addr).await {
            Ok(client) => {
                let client = Arc::new(client);
                let commit_config = RoundCommitConfig {
                    model_id: 0,
                    min_proofs: 1,
                    collection_timeout: Duration::from_secs(120),
                    ..Default::default()
                };
                let manager = RoundCommitManager::with_client(commit_config, client);
                orchestrator.set_round_commit_manager(manager);
                info!("On-chain proof submission enabled");
            }
            Err(e) => {
                warn!("Failed to connect to chain: {}. Running without on-chain submission.", e);
            }
        }
    } else {
        info!("No chain config (HELIX_ETH_RPC, PRIVATE_KEY, COORDINATOR_ADDRESS). Running without on-chain submission.");
    }

    let mut event_rx = orchestrator.start().await;

    // Compute initial model hash
    let initial_model = helix_node::trainer::MlpModel::new_random(d_in, d_hid, d_out, model_seed);
    let model_hash = initial_model.commitment();
    info!(
        "Model initialized: {}x{}x{} ({} params), commitment={}",
        d_in,
        d_hid,
        d_out,
        initial_model.num_params(),
        hex::encode(&model_hash[..8]),
    );

    // Set up shared API state
    let (round_trigger_tx, mut round_trigger_rx) = broadcast::channel::<()>(16);
    let api_snapshot = Arc::new(RwLock::new(OrchestratorSnapshot::default()));
    let proof_status = Arc::new(RwLock::new(Vec::<ProofStatusEntry>::new()));
    let api_peers = Arc::new(RwLock::new(PeerSnapshot::default()));
    let api_metrics = Arc::new(RwLock::new(MetricsSnapshot::default()));
    let api_key = helix_node::config::ApiConfig::default().api_key_or_generate();
    log::info!("API key: {}", api_key);
    let api_state = Arc::new(ApiState {
        orchestrator_workers: api_snapshot.clone(),
        round_trigger_tx: round_trigger_tx.clone(),
        peers: api_peers.clone(),
        metrics: api_metrics.clone(),
        api_key,
        rate_limiter: Arc::new(ApiRateLimiter::new(100)),
    });

    // Start HTTP API
    let http_addr: SocketAddr = format!("0.0.0.0:{}", http_port).parse().unwrap();
    let api_state_clone = api_state.clone();
    tokio::spawn(async move {
        if let Err(e) = helix_node::api::http::start_api_server(http_addr, api_state_clone).await {
            error!("HTTP API error: {}", e);
        }
    });

    // Start JSON-RPC server
    let (stop_trigger_tx_a, _) = broadcast::channel::<()>(16);
    let rpc_state = Arc::new(RpcState {
        snapshot: api_snapshot.clone(),
        round_trigger_tx: round_trigger_tx.clone(),
        stop_trigger_tx: stop_trigger_tx_a,
        node_role: "aggregator".to_string(),
        start_time: std::time::Instant::now(),
        proof_status: proof_status.clone(),
        proof_queue: Arc::new(RwLock::new(Vec::new())),
        node_config: Arc::new(RwLock::new(NodeConfigSnapshot::default())),
        aggregation_results: Arc::new(RwLock::new(Vec::new())),
        mpc_status: Arc::new(RwLock::new(MPCStatusSnapshot::default())),
        model_id: Arc::new(RwLock::new(None)),
        rpc_addr: format!("0.0.0.0:{}", rpc_port),
        rate_limiter: Arc::new(RwLock::new(RpcRateLimiter::default())),
    });
    let rpc_addr: SocketAddr = format!("0.0.0.0:{}", rpc_port).parse().unwrap();
    let rpc_state_clone = rpc_state.clone();
    tokio::spawn(async move {
        if let Err(e) = start_rpc_server(rpc_addr, rpc_state_clone).await {
            error!("JSON-RPC server error: {}", e);
        }
    });

    let orchestrator_ref = &orchestrator;

    // Spawn event logger + snapshot updater
    let snapshot_for_events = api_snapshot.clone();
    let proof_status_events = proof_status.clone();
    let mut completed_rounds = 0u64;
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match &event {
                OrchestratorEvent::WorkerJoined { peer_id } => {
                    info!("Worker joined: {}", peer_id);
                }
                OrchestratorEvent::WorkerLeft { peer_id } => {
                    info!("Worker left: {}", peer_id);
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
                    info!("Gradient received for round {} from {}", round_id, peer_id);
                    // Update proof collection count
                    let mut status = proof_status_events.write();
                    if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                        entry.proofs_collected += 1;
                    }
                }
                OrchestratorEvent::RoundCompleted { round_id, result_hash } => {
                    completed_rounds += 1;
                    snapshot_for_events.write().completed_rounds = completed_rounds;
                    // Update proof status
                    let mut status = proof_status_events.write();
                    if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                        entry.status = "completed".to_string();
                    }
                    info!(
                        "Round {} completed, result={}",
                        round_id,
                        hex::encode(&result_hash[..8])
                    );
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

    // Polling loop with graceful shutdown
    let mut round_number = 0u64;

    tokio::select! {
        _ = shutdown_signal() => {
            info!("Aggregator shutting down...");
            // Flush pending proofs
            let pending = proof_status.read().iter()
                .filter(|e| e.status == "collecting")
                .count();
            if pending > 0 {
                warn!("Shutting down with {} pending proof collections", pending);
            }
        }
        _ = async {
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;

                let stats = orchestrator_ref.worker_stats();

                // Update API snapshot
                {
                    let mut snap = api_snapshot.write();
                    snap.worker_count = stats.total;
                    snap.available_workers = stats.available;
                    snap.computing_workers = stats.computing;
                    snap.current_round = orchestrator_ref.current_round().map(|(id, phase)| RoundInfo {
                        round_id: id,
                        phase: format!("{:?}", phase),
                        gradients_received: 0,
                        workers_assigned: stats.computing,
                        commitment_hash: None,
                    });
                }

                info!(
                    "Workers: total={}, available={}, computing={}",
                    stats.total, stats.available, stats.computing
                );

                // Check for manual round trigger from HTTP API or RPC
                let manual_trigger = round_trigger_rx.try_recv().is_ok();

                // Start a new round if we have enough workers and no active round
                if (stats.available >= min_workers || manual_trigger) && orchestrator_ref.current_round().is_none() {
                    if stats.available < 1 {
                        continue;
                    }
                    round_number += 1;
                    info!("Starting round {} with {} available workers", round_number, stats.available);
                    match orchestrator_ref.start_round(model_hash).await {
                        Ok(id) => info!("Round {} started successfully (id={})", round_number, id),
                        Err(e) => error!("Failed to start round: {}", e),
                    }
                }
            }
        } => {}
    }

    info!("Aggregator shutdown complete (rounds_completed={})", round_number);
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Generates deterministic synthetic training data from a seed.
fn generate_training_data(d_in: usize, d_out: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = seed;
    let mut next = || -> f64 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };

    let x: Vec<f64> = (0..d_in).map(|_| next() * 0.5).collect();
    let target: Vec<f64> = (0..d_out).map(|_| next().abs()).collect();

    (x, target)
}
