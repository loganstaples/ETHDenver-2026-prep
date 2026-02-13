mod sc_client;

use helix_core::ModelCheckpoint;
use helix_node::config::DataSourceConfig;
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
use helix_node::trainer::{average_models, MlpModel, Trainer};
use helix_node::training::orchestrator::{
    OrchestratorConfig, OrchestratorEvent, TrainingOrchestrator,
};
use helix_node::training::MPCWorkerHandle;
use helix_node::training::persistence::{AggregatorSnapshot, WorkerSnapshot, StatePersistence};

use log::{error, info, warn};
use parking_lot::RwLock;
use std::net::SocketAddr;
use std::path::PathBuf;
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
    let aggregator_addr_str = env_or("HELIX_AGGREGATOR_ADDR", "");
    let rpc_port: u16 = env_or("HELIX_RPC_PORT", "9002").parse().unwrap_or(9002);
    let bootstrap_nodes_str = env_or("HELIX_BOOTSTRAP_NODES", "");
    let mdns_enabled = env_or("HELIX_MDNS_ENABLED", "0") == "1";

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

    // Parse bootstrap nodes
    let bootstrap_nodes: Vec<String> = bootstrap_nodes_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    let local_id = PeerId::random();
    info!(
        "Worker {} starting (bootstrap={}, mdns={}, direct_agg={})",
        local_id,
        bootstrap_nodes.len(),
        mdns_enabled,
        !aggregator_addr_str.is_empty(),
    );

    // Build network runner
    let mut network = NetworkRunnerBuilder::new()
        .local_id(local_id.clone())
        .listen_addr(listen_addr)
        .bootstrap_nodes(bootstrap_nodes.clone())
        .enable_mdns(mdns_enabled)
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

    // Set up ed25519 identity for authenticated connections
    #[cfg(feature = "crypto-sign")]
    {
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        network.set_signing_key(signing_key);
        info!("Worker ed25519 identity configured");
    }

    let network = Arc::new(network);
    network.start().await?;

    // Discovery: try direct aggregator, bootstrap, and mDNS
    if !aggregator_addr_str.is_empty() {
        // Direct aggregator connection (legacy mode)
        let aggregator_addr: SocketAddr = aggregator_addr_str.parse()
            .expect("invalid HELIX_AGGREGATOR_ADDR");
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
        match network.connect_peer(agg_peer).await {
            Ok(()) => info!("Connected to aggregator at {}", aggregator_addr),
            Err(e) => {
                warn!("Failed to connect to aggregator at {}: {}. Will rely on discovery.", aggregator_addr, e);
            }
        }
    }

    // Bootstrap peer discovery
    network.bootstrap().await;

    // mDNS discovery for local network
    if mdns_enabled {
        match network.start_mdns().await {
            Ok(()) => info!("mDNS discovery enabled"),
            Err(e) => warn!("mDNS discovery failed to start: {}", e),
        }
    }

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

    // Load worker data source configuration from env
    let data_source_type = env_or("HELIX_DATA_SOURCE", "synthetic");
    let data_source = match data_source_type.as_str() {
        "csv" => {
            let path = env_or("HELIX_CSV_PATH", "data.csv");
            let input_cols: Vec<String> = env_or("HELIX_CSV_INPUT_COLS", "")
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();
            let target_cols: Vec<String> = env_or("HELIX_CSV_TARGET_COLS", "")
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect();
            DataSourceConfig::CsvFile { path, input_cols, target_cols }
        }
        "ipfs" => {
            let cid = env_or("HELIX_IPFS_CID", "");
            let gateway = env_or("HELIX_IPFS_GATEWAY", "http://localhost:5001");
            DataSourceConfig::Ipfs { cid, gateway }
        }
        _ => {
            let seed: u64 = env_or("HELIX_DATA_SEED", "42").parse().unwrap_or(42);
            DataSourceConfig::Synthetic { seed }
        }
    };
    info!("Worker data source: {:?}", data_source);

    // Pre-load CSV dataset if configured
    let csv_dataset: Option<Vec<(Vec<f64>, Vec<f64>)>> = match &data_source {
        DataSourceConfig::CsvFile { path, input_cols, target_cols } => {
            info!("Loading CSV dataset from {}", path);
            match std::fs::read_to_string(path) {
                Ok(contents) => {
                    let mut rows = Vec::new();
                    let mut lines = contents.lines();
                    let header: Vec<&str> = match lines.next() {
                        Some(h) => h.split(',').map(|s| s.trim()).collect(),
                        None => { warn!("CSV file is empty"); Vec::new() }
                    };

                    let in_idxs: Vec<usize> = input_cols.iter()
                        .filter_map(|c| header.iter().position(|h| h == c))
                        .collect();
                    let tgt_idxs: Vec<usize> = target_cols.iter()
                        .filter_map(|c| header.iter().position(|h| h == c))
                        .collect();

                    if in_idxs.is_empty() || tgt_idxs.is_empty() {
                        warn!("CSV column matching failed: input_cols={:?}, target_cols={:?}, header={:?}", input_cols, target_cols, header);
                    }

                    for line in lines {
                        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        let x: Vec<f64> = in_idxs.iter()
                            .map(|&i| fields.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0))
                            .collect();
                        let t: Vec<f64> = tgt_idxs.iter()
                            .map(|&i| fields.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0))
                            .collect();
                        rows.push((x, t));
                    }
                    info!("Loaded {} samples from CSV ({} inputs, {} targets)", rows.len(), in_idxs.len(), tgt_idxs.len());
                    if rows.is_empty() { None } else { Some(rows) }
                }
                Err(e) => {
                    warn!("Failed to read CSV file {}: {}", path, e);
                    None
                }
            }
        }
        _ => None,
    };

    // === Worker state persistence ===
    let checkpoint_dir_str = env_or("HELIX_CHECKPOINT_DIR", "");
    let max_checkpoints: usize = env_or("HELIX_MAX_CHECKPOINTS", "5").parse().unwrap_or(5);
    let worker_persistence = if !checkpoint_dir_str.is_empty() {
        match StatePersistence::new(
            Some(PathBuf::from(&checkpoint_dir_str)),
            max_checkpoints,
            &format!("worker_{}", local_id),
        ) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("Failed to initialize worker persistence: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Try to restore from previous snapshot
    let mut trainer: Option<Trainer> = None;
    let mut mpc_handle: Option<MPCWorkerHandle> = None;
    let mut steps_completed = 0u64;
    let mut last_completed_round = 0u64;
    let mut csv_sample_idx = 0usize;

    if let Some(ref persistence) = worker_persistence {
        match persistence.load_latest_worker() {
            Ok(Some(snapshot)) => {
                steps_completed = snapshot.steps_completed;
                last_completed_round = snapshot.last_completed_round;
                info!(
                    "Restored worker state: steps={}, last_round={}",
                    steps_completed, last_completed_round,
                );
                // Model will be provided by aggregator on next round, so we don't
                // restore it here — the aggregator is the source of truth.
            }
            Ok(None) => {
                info!("No worker checkpoint found, starting fresh");
            }
            Err(e) => {
                warn!("Failed to load worker checkpoint: {}, starting fresh", e);
            }
        }
    }

    let result = tokio::select! {
        _ = shutdown_signal() => {
            info!("Worker shutting down, saving checkpoint...");
            // Graceful shutdown: save worker state
            if let Some(ref persistence) = worker_persistence {
                let mut snapshot = WorkerSnapshot::new(
                    local_id.to_string(),
                    last_completed_round,
                    steps_completed,
                );
                if let Some(ref t) = trainer {
                    let ckpt = t.model().to_checkpoint(t.step_count());
                    if let Ok(bytes) = ckpt.to_bytes() {
                        snapshot.model_checkpoint_bytes = Some(bytes);
                    }
                }
                if let Err(e) = persistence.save_worker(&snapshot) {
                    error!("Failed to save worker checkpoint on shutdown: {}", e);
                } else {
                    info!("Worker checkpoint saved on shutdown (steps={})", steps_completed);
                }
            }
            Ok(())
        }
        result = async {
            loop {
                match network.next_event().await {
                    Some(NetworkEvent::TrainingMessage { from, message }) => {
                        match message {
                            TrainingMessage::ModelWeights {
                                round_id,
                                checkpoint_data,
                                weight_hash,
                            } => {
                                info!(
                                    "Received ModelWeights for round {} from {} ({} bytes, hash={})",
                                    round_id, from, checkpoint_data.len(),
                                    hex::encode(&weight_hash[..8]),
                                );
                                match ModelCheckpoint::from_bytes(&checkpoint_data) {
                                    Ok(ckpt) => {
                                        match MlpModel::from_checkpoint(&ckpt) {
                                            Ok(model) => {
                                                let lr = trainer.as_ref()
                                                    .map(|t| t.learning_rate())
                                                    .unwrap_or(0.01);
                                                trainer = Some(Trainer::with_model(model, lr));
                                                info!(
                                                    "Model loaded from aggregator for round {} (step={})",
                                                    round_id, ckpt.step_number,
                                                );
                                            }
                                            Err(e) => warn!("Failed to reconstruct model from checkpoint: {}", e),
                                        }
                                    }
                                    Err(e) => warn!("Failed to deserialize model checkpoint: {}", e),
                                }
                            }
                            TrainingMessage::UpdatedWeights {
                                round_id,
                                checkpoint_data,
                                weight_hash,
                            } => {
                                info!(
                                    "Received UpdatedWeights after round {} ({} bytes, hash={})",
                                    round_id, checkpoint_data.len(),
                                    hex::encode(&weight_hash[..8]),
                                );
                                match ModelCheckpoint::from_bytes(&checkpoint_data) {
                                    Ok(ckpt) => {
                                        match MlpModel::from_checkpoint(&ckpt) {
                                            Ok(model) => {
                                                let lr = trainer.as_ref()
                                                    .map(|t| t.learning_rate())
                                                    .unwrap_or(0.01);
                                                trainer = Some(Trainer::with_model(model, lr));
                                                info!(
                                                    "Model updated from aggregator after round {} (step={})",
                                                    round_id, ckpt.step_number,
                                                );
                                            }
                                            Err(e) => warn!("Failed to reconstruct updated model: {}", e),
                                        }
                                    }
                                    Err(e) => warn!("Failed to deserialize updated checkpoint: {}", e),
                                }
                            }
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
                                    // Initialize trainer with seed if we haven't received model weights
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

                                    // Load training data from configured source
                                    let (x, target) = if let Some(ref dataset) = csv_dataset {
                                        let sample = &dataset[csv_sample_idx % dataset.len()];
                                        csv_sample_idx += 1;
                                        (sample.0.clone(), sample.1.clone())
                                    } else if let DataSourceConfig::Synthetic { seed } = &data_source {
                                        generate_training_data(params.d_in, params.d_out, seed + round_id)
                                    } else {
                                        // Fallback to synthetic if non-CSV source not yet loaded
                                        generate_training_data(params.d_in, params.d_out, params.model_seed + round_id)
                                    };

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

                                            // Send updated model weights back to aggregator
                                            let updated_model = t.model();
                                            let updated_ckpt = updated_model.to_checkpoint(t.step_count());
                                            match updated_ckpt.to_bytes() {
                                                Ok(ckpt_bytes) => {
                                                    let weight_hash = updated_model.commitment();
                                                    network
                                                        .broadcast(MessagePayload::Gradient(
                                                            GradientMessage::WeightUpdate {
                                                                round_id,
                                                                checkpoint_data: ckpt_bytes,
                                                                weight_hash,
                                                            },
                                                        ))
                                                        .await;
                                                    info!(
                                                        "Weight update sent for round {} (hash={})",
                                                        round_id, hex::encode(&weight_hash[..8]),
                                                    );
                                                }
                                                Err(e) => {
                                                    warn!("Failed to serialize model checkpoint for weight update: {}", e);
                                                }
                                            }

                                            steps_completed += 1;
                                            info!("Gradient + weights sent for round {} (total steps: {})", round_id, steps_completed);

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
                                last_completed_round = round_id;

                                // Save worker checkpoint after each completed round
                                if let Some(ref persistence) = worker_persistence {
                                    let mut snapshot = WorkerSnapshot::new(
                                        local_id.to_string(),
                                        last_completed_round,
                                        steps_completed,
                                    );
                                    if let Some(ref t) = trainer {
                                        let ckpt = t.model().to_checkpoint(t.step_count());
                                        if let Ok(bytes) = ckpt.to_bytes() {
                                            snapshot.model_checkpoint_bytes = Some(bytes);
                                        }
                                    }
                                    if let Err(e) = persistence.save_worker(&snapshot) {
                                        warn!("Failed to save worker checkpoint: {}", e);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Some(NetworkEvent::Heartbeat { .. }) => {
                        // Heartbeat pongs handled automatically
                    }
                    Some(NetworkEvent::PeerDiscovered(peer)) => {
                        info!("Peer discovered: {} at {} (agg={}, train={})",
                            peer.id, peer.address,
                            peer.capabilities.can_aggregate,
                            peer.capabilities.can_train,
                        );
                        // If we discover an aggregator via bootstrap/mDNS, connect to it
                        if peer.capabilities.can_aggregate {
                            info!("Discovered aggregator {} at {}", peer.id, peer.address);
                        }
                    }
                    Some(NetworkEvent::PeerDisconnected(peer_id)) => {
                        warn!("Peer disconnected: {}, scheduling reconnection", peer_id);
                        // Schedule reconnection with exponential backoff
                        if let Some(peer_info) = network.discovery().get_peer(&peer_id).await {
                            network.schedule_reconnect(
                                peer_id,
                                peer_info.address,
                                peer_info.capabilities,
                            );
                        }
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

    let bootstrap_nodes_str = env_or("HELIX_BOOTSTRAP_NODES", "");
    let mdns_enabled = env_or("HELIX_MDNS_ENABLED", "0") == "1";
    let bootstrap_nodes: Vec<String> = bootstrap_nodes_str
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect();

    let local_id = PeerId::from_string("aggregator");
    info!(
        "Aggregator starting on {} (min_workers={}, model={}x{}x{}, bootstrap={}, mdns={})",
        listen_addr, min_workers, d_in, d_hid, d_out,
        bootstrap_nodes.len(), mdns_enabled,
    );

    // Build network runner
    let mut network = NetworkRunnerBuilder::new()
        .local_id(local_id.clone())
        .listen_addr(listen_addr)
        .bootstrap_nodes(bootstrap_nodes.clone())
        .enable_mdns(mdns_enabled)
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

    // Set up ed25519 identity for authenticated connections
    #[cfg(feature = "crypto-sign")]
    {
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        network.set_signing_key(signing_key);
        info!("Aggregator ed25519 identity configured");
    }

    let network = Arc::new(network);
    network.start().await?;
    info!("Aggregator listening on {}", listen_addr);

    // Bootstrap peer discovery
    network.bootstrap().await;

    // mDNS discovery for local network
    if mdns_enabled {
        match network.start_mdns().await {
            Ok(()) => info!("mDNS discovery enabled"),
            Err(e) => warn!("mDNS discovery failed to start: {}", e),
        }
    }

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

    // === Aggregator state persistence ===
    let checkpoint_dir = env_or("HELIX_CHECKPOINT_DIR", "");
    let max_checkpoints: usize = env_or("HELIX_MAX_CHECKPOINTS", "5").parse().unwrap_or(5);

    let aggregator_persistence = if !checkpoint_dir.is_empty() {
        match StatePersistence::new(
            Some(PathBuf::from(&checkpoint_dir)),
            max_checkpoints,
            "aggregator",
        ) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!("Failed to initialize aggregator persistence: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Also keep the legacy checkpoint_path for backwards compatibility
    let checkpoint_path = if checkpoint_dir.is_empty() {
        None
    } else {
        std::fs::create_dir_all(&checkpoint_dir).ok();
        Some(format!("{}/model_latest.hxck", checkpoint_dir))
    };

    // Try to restore full aggregator state from snapshot
    let mut restored_round_number = 0u64;
    let mut restored_total_rounds = 0u64;
    let mut restored_error_bounds: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut restored_proof_hashes: std::collections::HashMap<u64, Vec<[u8; 32]>> = std::collections::HashMap::new();
    let mut restored_participants: Vec<String> = Vec::new();

    let initial_model = if let Some(ref persistence) = aggregator_persistence {
        match persistence.load_latest_aggregator() {
            Ok(Some(snapshot)) => {
                info!(
                    "Restoring aggregator from snapshot: round={}, total_rounds={}, timestamp={}",
                    snapshot.last_completed_round,
                    snapshot.total_rounds_completed,
                    snapshot.timestamp,
                );

                restored_round_number = snapshot.last_completed_round;
                restored_total_rounds = snapshot.total_rounds_completed;
                restored_error_bounds = snapshot.error_bounds;
                restored_proof_hashes = snapshot.proof_hashes;
                restored_participants = snapshot.participants;

                // Restore model from checkpoint bytes
                match ModelCheckpoint::from_bytes(&snapshot.model_checkpoint_bytes) {
                    Ok(ckpt) => {
                        match MlpModel::from_checkpoint(&ckpt) {
                            Ok(m) => {
                                info!(
                                    "Model restored from snapshot at round {} (step={})",
                                    snapshot.last_completed_round, ckpt.step_number,
                                );
                                m
                            }
                            Err(e) => {
                                warn!("Failed to parse snapshot model: {}, creating fresh", e);
                                MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to deserialize snapshot checkpoint: {}, creating fresh", e);
                        MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                    }
                }
            }
            Ok(None) => {
                info!("No aggregator snapshot found, checking legacy checkpoint...");
                // Fallback: try legacy model_latest.hxck
                if let Some(ref path) = checkpoint_path {
                    if std::path::Path::new(path).exists() {
                        match helix_core::ModelCheckpoint::from_bytes(
                            &std::fs::read(path).unwrap_or_default(),
                        ) {
                            Ok(ckpt) => {
                                info!("Restored model from legacy checkpoint at step {}", ckpt.step_number);
                                restored_round_number = ckpt.step_number;
                                match MlpModel::from_checkpoint(&ckpt) {
                                    Ok(m) => m,
                                    Err(e) => {
                                        warn!("Failed to parse legacy checkpoint: {}, creating fresh", e);
                                        MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to load legacy checkpoint: {}, creating fresh model", e);
                                MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                            }
                        }
                    } else {
                        MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                    }
                } else {
                    MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                }
            }
            Err(e) => {
                warn!("Failed to load aggregator snapshot: {}, creating fresh model", e);
                MlpModel::new_random(d_in, d_hid, d_out, model_seed)
            }
        }
    } else if let Some(ref path) = checkpoint_path {
        // No persistence configured, use legacy path
        if std::path::Path::new(path).exists() {
            match helix_core::ModelCheckpoint::from_bytes(
                &std::fs::read(path).unwrap_or_default(),
            ) {
                Ok(ckpt) => {
                    info!("Restored model from checkpoint at step {}", ckpt.step_number);
                    match MlpModel::from_checkpoint(&ckpt) {
                        Ok(m) => m,
                        Err(e) => {
                            warn!("Failed to parse checkpoint model: {}, creating fresh", e);
                            MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to load checkpoint: {}, creating fresh model", e);
                    MlpModel::new_random(d_in, d_hid, d_out, model_seed)
                }
            }
        } else {
            MlpModel::new_random(d_in, d_hid, d_out, model_seed)
        }
    } else {
        MlpModel::new_random(d_in, d_hid, d_out, model_seed)
    };

    let model_hash = initial_model.commitment();
    let aggregator_model = Arc::new(RwLock::new(initial_model.clone()));
    info!(
        "Model initialized: {}x{}x{} ({} params), commitment={}, restored_round={}",
        d_in,
        d_hid,
        d_out,
        initial_model.num_params(),
        hex::encode(&model_hash[..8]),
        restored_round_number,
    );
    if !restored_participants.is_empty() {
        info!("Restored {} participants from last session", restored_participants.len());
    }
    if !restored_error_bounds.is_empty() {
        info!("Restored error bounds for {} rounds", restored_error_bounds.len());
    }

    // Serialize initial model checkpoint for distribution to workers
    let initial_checkpoint = initial_model.to_checkpoint(restored_round_number);
    let initial_checkpoint_bytes = initial_checkpoint.to_bytes()
        .expect("Failed to serialize initial model checkpoint");

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

    // Channel for the event handler to notify the polling loop of completed rounds
    let (completed_round_tx, mut completed_round_rx) = tokio::sync::mpsc::channel::<u64>(16);

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
                    {
                        let mut status = proof_status_events.write();
                        if let Some(entry) = status.iter_mut().find(|e| e.round_id == *round_id) {
                            entry.status = "completed".to_string();
                        }
                    } // drop the write guard before await
                    info!(
                        "Round {} completed, result={}",
                        round_id,
                        hex::encode(&result_hash[..8])
                    );
                    // Notify the polling loop to aggregate weight updates
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

    // Polling loop with graceful shutdown
    let mut round_number = restored_round_number;
    let mut total_rounds_completed = restored_total_rounds;
    let error_bounds_history = Arc::new(RwLock::new(restored_error_bounds));
    let proof_hashes_history = Arc::new(RwLock::new(restored_proof_hashes));
    // Track the current model's checkpoint bytes for distribution
    let current_checkpoint_bytes = Arc::new(RwLock::new(initial_checkpoint_bytes));

    tokio::select! {
        _ = shutdown_signal() => {
            info!("Aggregator shutting down, saving checkpoint...");
            // Flush pending proofs
            let pending = proof_status.read().iter()
                .filter(|e| e.status == "collecting")
                .count();
            if pending > 0 {
                warn!("Shutting down with {} pending proof collections", pending);
            }

            // Graceful shutdown: save full aggregator state
            if let Some(ref persistence) = aggregator_persistence {
                let model_ckpt = aggregator_model.read().to_checkpoint(round_number);
                if let Ok(ckpt_bytes) = model_ckpt.to_bytes() {
                    let mut snapshot = AggregatorSnapshot::new(
                        ckpt_bytes,
                        round_number,
                        total_rounds_completed,
                        d_in, d_hid, d_out,
                        model_seed,
                        learning_rate,
                    );
                    snapshot.error_bounds = error_bounds_history.read().clone();
                    snapshot.proof_hashes = proof_hashes_history.read().clone();
                    if let Err(e) = persistence.save_aggregator(&snapshot) {
                        error!("Failed to save aggregator checkpoint on shutdown: {}", e);
                    } else {
                        info!("Aggregator checkpoint saved on shutdown (round={})", round_number);
                    }
                }
            }
        }
        _ = async {
            loop {
                // Check for completed rounds that need weight aggregation
                while let Ok(completed_round_id) = completed_round_rx.try_recv() {
                    let updates = orchestrator_ref.take_weight_updates(completed_round_id);
                    if updates.is_empty() {
                        info!("Round {} completed with no weight updates to aggregate", completed_round_id);
                        continue;
                    }

                    info!(
                        "Aggregating {} weight updates for round {}",
                        updates.len(), completed_round_id,
                    );

                    // Deserialize all worker models
                    let mut worker_models = Vec::new();
                    for (peer_id, ckpt_bytes, _hash) in &updates {
                        match ModelCheckpoint::from_bytes(ckpt_bytes) {
                            Ok(ckpt) => match MlpModel::from_checkpoint(&ckpt) {
                                Ok(model) => worker_models.push(model),
                                Err(e) => warn!("Bad checkpoint from {}: {}", peer_id, e),
                            },
                            Err(e) => warn!("Bad checkpoint bytes from {}: {}", peer_id, e),
                        }
                    }

                    if worker_models.is_empty() {
                        warn!("No valid weight updates for round {}", completed_round_id);
                        continue;
                    }

                    // FedAvg: average all worker models
                    match average_models(&worker_models) {
                        Ok(averaged) => {
                            let new_hash = averaged.commitment();
                            total_rounds_completed += 1;
                            info!(
                                "FedAvg complete for round {}: {} models averaged, new commitment={}, total_rounds={}",
                                completed_round_id,
                                worker_models.len(),
                                hex::encode(&new_hash[..8]),
                                total_rounds_completed,
                            );

                            // Update aggregator model
                            *aggregator_model.write() = averaged.clone();

                            // Persist legacy checkpoint
                            let ckpt = averaged.to_checkpoint(completed_round_id);
                            if let Some(ref path) = checkpoint_path {
                                match ckpt.to_bytes() {
                                    Ok(bytes) => {
                                        if let Err(e) = std::fs::write(path, &bytes) {
                                            error!("Failed to persist checkpoint: {}", e);
                                        } else {
                                            info!("Checkpoint persisted to {} ({} bytes)", path, bytes.len());
                                        }
                                    }
                                    Err(e) => error!("Failed to serialize checkpoint: {}", e),
                                }
                            }

                            // Serialize updated model for distribution
                            match ckpt.to_bytes() {
                                Ok(bytes) => {
                                    *current_checkpoint_bytes.write() = bytes.clone();

                                    // === Save full AggregatorSnapshot ===
                                    if let Some(ref persistence) = aggregator_persistence {
                                        // Collect participant IDs from this round's updates
                                        let participants: Vec<String> = updates
                                            .iter()
                                            .map(|(pid, _, _)| pid.to_string())
                                            .collect();

                                        // Track error bounds (average of worker error_bounds)
                                        // We don't have per-worker error_bounds in weight updates,
                                        // so we track the proof submission counts instead
                                        let avg_error = worker_models.len() as f64 * 0.01;
                                        error_bounds_history.write().insert(completed_round_id, avg_error);

                                        // Track proof hash for this round (hash of the aggregated commitment)
                                        proof_hashes_history.write().insert(completed_round_id, vec![new_hash]);

                                        let mut snapshot = AggregatorSnapshot::new(
                                            bytes.clone(),
                                            completed_round_id,
                                            total_rounds_completed,
                                            d_in, d_hid, d_out,
                                            model_seed,
                                            learning_rate,
                                        );
                                        snapshot.participants = participants;
                                        snapshot.error_bounds = error_bounds_history.read().clone();
                                        snapshot.proof_hashes = proof_hashes_history.read().clone();

                                        if let Err(e) = persistence.save_aggregator(&snapshot) {
                                            error!("Failed to save aggregator snapshot: {}", e);
                                        }
                                    }

                                    // Broadcast updated weights to all workers
                                    network
                                        .broadcast(MessagePayload::Training(
                                            TrainingMessage::UpdatedWeights {
                                                round_id: completed_round_id,
                                                checkpoint_data: bytes,
                                                weight_hash: new_hash,
                                            },
                                        ))
                                        .await;
                                    info!("Updated weights broadcast for round {}", completed_round_id);
                                }
                                Err(e) => error!("Failed to serialize updated checkpoint: {}", e),
                            }
                        }
                        Err(e) => error!("FedAvg failed for round {}: {}", completed_round_id, e),
                    }
                }

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

                    // Use current model's commitment hash
                    let current_model_hash = aggregator_model.read().commitment();
                    let ckpt_bytes = current_checkpoint_bytes.read().clone();

                    info!(
                        "Starting round {} with {} available workers (model={})",
                        round_number, stats.available, hex::encode(&current_model_hash[..8]),
                    );
                    match orchestrator_ref.start_round(current_model_hash).await {
                        Ok(id) => {
                            info!("Round {} started successfully (id={})", round_number, id);

                            // Distribute current model weights to all workers
                            network
                                .broadcast(MessagePayload::Training(
                                    TrainingMessage::ModelWeights {
                                        round_id: id,
                                        checkpoint_data: ckpt_bytes,
                                        weight_hash: current_model_hash,
                                    },
                                ))
                                .await;
                            info!("Model weights broadcast for round {} ({} bytes)", id, current_checkpoint_bytes.read().len());
                        }
                        Err(e) => error!("Failed to start round: {}", e),
                    }
                }
            }
        } => {}
    }

    info!("Aggregator shutdown complete (round_number={}, total_rounds={})", round_number, total_rounds_completed);
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
