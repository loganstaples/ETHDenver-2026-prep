//! HELIX Network Orchestrator
//!
//! Manages multi-node networks for demos and testing.
//! Supports both simulated nodes and real process spawning.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, broadcast};

/// Network orchestrator for managing multi-node deployments
pub struct NetworkOrchestrator {
    /// Network name
    name: String,
    /// Network configuration
    config: NetworkOrchestratorConfig,
    /// Running nodes
    nodes: Arc<RwLock<HashMap<String, NodeHandle>>>,
    /// Shutdown channel
    shutdown_tx: broadcast::Sender<()>,
    /// Live mode configuration (None = simulated)
    live_config: Option<LiveNetworkConfig>,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOrchestratorConfig {
    /// Number of worker nodes
    pub worker_count: u32,
    /// Number of aggregator nodes
    pub aggregator_count: u32,
    /// Base port for nodes
    pub base_port: u16,
    /// Data directory
    pub data_dir: PathBuf,
    /// RPC endpoint
    pub rpc_url: String,
    /// Network type (local, testnet, mainnet)
    pub network_type: String,
}

impl Default for NetworkOrchestratorConfig {
    fn default() -> Self {
        Self {
            worker_count: 3,
            aggregator_count: 1,
            base_port: 9000,
            data_dir: PathBuf::from("./helix-demo"),
            rpc_url: "http://localhost:8545".to_string(),
            network_type: "local".to_string(),
        }
    }
}

/// Configuration for spawning real helix-node processes.
#[derive(Debug, Clone)]
pub struct LiveNetworkConfig {
    /// Path to the helix-node binary.
    pub node_binary: PathBuf,
    /// Ethereum RPC URL.
    pub eth_rpc: String,
    /// Private key for on-chain transactions.
    pub private_key: String,
    /// Deployed coordinator contract address.
    pub coordinator_address: String,
    /// Model dimensions: (d_in, d_hid, d_out).
    pub model_dims: (usize, usize, usize),
    /// Model seed.
    pub model_seed: u64,
    /// Learning rate.
    pub learning_rate: f64,
    /// HTTP API port for aggregator.
    pub http_port: u16,
}

/// Handle to a running node
pub struct NodeHandle {
    /// Node ID
    pub id: String,
    /// Node role (worker, aggregator)
    pub role: NodeRole,
    /// Node status
    pub status: NodeStatus,
    /// Port the node is listening on
    pub port: u16,
    /// Process handle (if managed)
    process: Option<Child>,
}

/// Node role in the network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Worker,
    Aggregator,
    Coordinator,
}

/// Node status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Starting,
    Running,
    Idle,
    Training,
    Syncing,
    Stopped,
    Failed,
}

impl NetworkOrchestrator {
    /// Create a new network orchestrator
    pub fn new(name: &str, config: NetworkOrchestratorConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);

        Self {
            name: name.to_string(),
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx,
            live_config: None,
        }
    }

    /// Create a new orchestrator with live process spawning.
    pub fn new_live(name: &str, config: NetworkOrchestratorConfig, live: LiveNetworkConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);

        Self {
            name: name.to_string(),
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx,
            live_config: Some(live),
        }
    }

    /// Start the network
    pub async fn start(&self) -> Result<()> {
        tracing::info!("Starting network: {}", self.name);

        // Create data directory
        std::fs::create_dir_all(&self.config.data_dir)?;

        // Start aggregator nodes
        for i in 0..self.config.aggregator_count {
            let node_id = format!("helix-agg-{:02}", i + 1);
            let port = self.config.base_port + i as u16;

            self.start_node(&node_id, NodeRole::Aggregator, port).await?;
        }

        // Start worker nodes
        for i in 0..self.config.worker_count {
            let node_id = format!("helix-worker-{:02}", i + 1);
            let port = self.config.base_port + self.config.aggregator_count as u16 + i as u16;

            self.start_node(&node_id, NodeRole::Worker, port).await?;
        }

        tracing::info!("Network {} started with {} nodes",
            self.name,
            self.config.aggregator_count + self.config.worker_count
        );

        Ok(())
    }

    /// Start a single node
    async fn start_node(&self, node_id: &str, role: NodeRole, port: u16) -> Result<()> {
        tracing::debug!("Starting node: {} (role: {:?}, port: {})", node_id, role, port);

        let process = match &self.live_config {
            Some(live) => self.spawn_real_node(node_id, role, port, live)?,
            None => None,
        };

        let handle = NodeHandle {
            id: node_id.to_string(),
            role,
            status: NodeStatus::Running,
            port,
            process,
        };

        let mut nodes = self.nodes.write().await;
        nodes.insert(node_id.to_string(), handle);

        Ok(())
    }

    /// Spawns a real helix-node process with appropriate env vars.
    fn spawn_real_node(
        &self,
        node_id: &str,
        role: NodeRole,
        port: u16,
        live: &LiveNetworkConfig,
    ) -> Result<Option<Child>> {
        let role_str = match role {
            NodeRole::Worker => "worker",
            NodeRole::Aggregator => "aggregator",
            NodeRole::Coordinator => "aggregator",
        };

        let listen_addr = format!("127.0.0.1:{}", port);
        let aggregator_port = self.config.base_port; // Aggregator is always base_port

        let mut cmd = Command::new(&live.node_binary);
        cmd.env("HELIX_NODE_ROLE", role_str)
            .env("HELIX_LISTEN_ADDR", &listen_addr)
            .env("HELIX_D_IN", live.model_dims.0.to_string())
            .env("HELIX_D_HID", live.model_dims.1.to_string())
            .env("HELIX_D_OUT", live.model_dims.2.to_string())
            .env("HELIX_MODEL_SEED", live.model_seed.to_string())
            .env("HELIX_LR", live.learning_rate.to_string())
            .env("RUST_LOG", "info");

        match role {
            NodeRole::Worker => {
                cmd.env("HELIX_AGGREGATOR_ADDR", format!("127.0.0.1:{}", aggregator_port));
            }
            NodeRole::Aggregator | NodeRole::Coordinator => {
                cmd.env("HELIX_MIN_WORKERS", self.config.worker_count.to_string())
                    .env("HELIX_HTTP_PORT", live.http_port.to_string());

                // On-chain config
                if !live.eth_rpc.is_empty() {
                    cmd.env("HELIX_ETH_RPC", &live.eth_rpc)
                        .env("PRIVATE_KEY", &live.private_key)
                        .env("COORDINATOR_ADDRESS", &live.coordinator_address);
                }
            }
        }

        // Pipe stdout/stderr for logging
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped());

        tracing::info!("Spawning {} ({}) on {}", node_id, role_str, listen_addr);
        let child = cmd.spawn().map_err(|e| {
            anyhow!("Failed to spawn {}: {}. Is helix-node built? Run: cargo build -p helix-node", node_id, e)
        })?;

        Ok(Some(child))
    }

    /// Stop the network
    pub async fn stop(&self) -> Result<()> {
        tracing::info!("Stopping network: {}", self.name);

        // Send shutdown signal
        let _ = self.shutdown_tx.send(());

        // Stop all nodes
        let mut nodes = self.nodes.write().await;
        for (id, handle) in nodes.iter_mut() {
            tracing::debug!("Stopping node: {}", id);
            handle.status = NodeStatus::Stopped;

            if let Some(mut process) = handle.process.take() {
                let _ = process.kill();
                let _ = process.wait(); // reap the child
            }
        }

        nodes.clear();
        tracing::info!("Network {} stopped", self.name);

        Ok(())
    }

    /// Get network status
    pub async fn status(&self) -> NetworkStatus {
        let nodes = self.nodes.read().await;

        let node_statuses: Vec<NodeStatusInfo> = nodes
            .values()
            .map(|h| NodeStatusInfo {
                id: h.id.clone(),
                role: h.role,
                status: h.status,
                port: h.port,
            })
            .collect();

        let running = node_statuses.iter().filter(|n| n.status == NodeStatus::Running).count();

        NetworkStatus {
            name: self.name.clone(),
            total_nodes: node_statuses.len(),
            running_nodes: running,
            nodes: node_statuses,
        }
    }

    /// Scale the network to the specified number of workers
    pub async fn scale(&self, worker_count: u32) -> Result<()> {
        let current = {
            let nodes = self.nodes.read().await;
            nodes.values().filter(|n| n.role == NodeRole::Worker).count() as u32
        };

        if worker_count > current {
            // Scale up
            for i in current..worker_count {
                let node_id = format!("helix-worker-{:02}", i + 1);
                let port = self.config.base_port + self.config.aggregator_count as u16 + i as u16;
                self.start_node(&node_id, NodeRole::Worker, port).await?;
            }
        } else if worker_count < current {
            // Scale down
            let mut nodes = self.nodes.write().await;
            let to_remove: Vec<String> = nodes
                .iter()
                .filter(|(_, n)| n.role == NodeRole::Worker)
                .take((current - worker_count) as usize)
                .map(|(id, _)| id.clone())
                .collect();

            for id in to_remove {
                if let Some(mut handle) = nodes.remove(&id) {
                    if let Some(mut process) = handle.process.take() {
                        let _ = process.kill();
                        let _ = process.wait();
                    }
                }
            }
        }

        Ok(())
    }

    /// Get a shutdown receiver
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Wait for network to be ready
    pub async fn wait_ready(&self, timeout: Duration) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            let status = self.status().await;

            if status.running_nodes == status.total_nodes && status.total_nodes > 0 {
                return Ok(());
            }

            if start.elapsed() > timeout {
                return Err(anyhow!("Timeout waiting for network to be ready"));
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Returns the aggregator's HTTP API URL (if live mode).
    pub fn aggregator_http_url(&self) -> Option<String> {
        self.live_config
            .as_ref()
            .map(|live| format!("http://127.0.0.1:{}", live.http_port))
    }
}

/// Network status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub name: String,
    pub total_nodes: usize,
    pub running_nodes: usize,
    pub nodes: Vec<NodeStatusInfo>,
}

/// Individual node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatusInfo {
    pub id: String,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub port: u16,
}

/// Demo orchestrator that simulates a full network
pub struct DemoOrchestrator {
    orchestrator: NetworkOrchestrator,
    /// Demo configuration
    demo_config: DemoConfig,
}

/// Demo configuration
#[derive(Debug, Clone)]
pub struct DemoConfig {
    pub worker_count: u32,
    pub round_count: u32,
    pub round_duration: Duration,
    pub headless: bool,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            worker_count: 3,
            round_count: 5,
            round_duration: Duration::from_secs(10),
            headless: false,
        }
    }
}

impl DemoOrchestrator {
    /// Create a new demo orchestrator
    pub fn new(config: DemoConfig) -> Self {
        let orch_config = NetworkOrchestratorConfig {
            worker_count: config.worker_count,
            aggregator_count: 1,
            ..Default::default()
        };

        Self {
            orchestrator: NetworkOrchestrator::new("helix-demo", orch_config),
            demo_config: config,
        }
    }

    /// Run the demo
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        // Start network
        self.orchestrator.start().await?;

        // Wait for ready
        self.orchestrator.wait_ready(Duration::from_secs(30)).await?;

        // Run training rounds
        for round in 0..self.demo_config.round_count {
            tokio::select! {
                _ = tokio::time::sleep(self.demo_config.round_duration) => {
                    tracing::info!("Round {} completed", round + 1);
                }
                _ = shutdown.recv() => {
                    tracing::info!("Demo interrupted");
                    break;
                }
            }
        }

        // Stop network
        self.orchestrator.stop().await?;

        Ok(())
    }
}

/// Live training orchestrator that spawns real nodes and monitors progress.
pub struct LiveTrainingOrchestrator {
    orchestrator: NetworkOrchestrator,
    /// Number of training rounds.
    pub rounds: u32,
    /// HTTP API base URL for polling aggregator.
    pub aggregator_url: String,
}

impl LiveTrainingOrchestrator {
    /// Create a new live training orchestrator.
    pub fn new(
        worker_count: u32,
        rounds: u32,
        live: LiveNetworkConfig,
    ) -> Self {
        let http_port = live.http_port;
        let orch_config = NetworkOrchestratorConfig {
            worker_count,
            aggregator_count: 1,
            ..Default::default()
        };

        let orchestrator = NetworkOrchestrator::new_live("helix-train", orch_config, live);

        Self {
            orchestrator,
            rounds,
            aggregator_url: format!("http://127.0.0.1:{}", http_port),
        }
    }

    /// Start all nodes.
    pub async fn start(&self) -> Result<()> {
        self.orchestrator.start().await
    }

    /// Stop all nodes.
    pub async fn stop(&self) -> Result<()> {
        self.orchestrator.stop().await
    }

    /// Poll the aggregator's /health endpoint.
    pub async fn poll_health(&self) -> Result<serde_json::Value> {
        let url = format!("{}/health", self.aggregator_url);
        let resp = reqwest::get(&url).await?.json::<serde_json::Value>().await?;
        Ok(resp)
    }

    /// Poll the aggregator's /round/status endpoint.
    pub async fn poll_round_status(&self) -> Result<serde_json::Value> {
        let url = format!("{}/round/status", self.aggregator_url);
        let resp = reqwest::get(&url).await?.json::<serde_json::Value>().await?;
        Ok(resp)
    }

    /// Trigger a round start via POST /round/start.
    pub async fn trigger_round(&self) -> Result<serde_json::Value> {
        let url = format!("{}/round/start", self.aggregator_url);
        let client = reqwest::Client::new();
        let resp = client.post(&url).send().await?.json::<serde_json::Value>().await?;
        Ok(resp)
    }

    /// Get network status.
    pub async fn status(&self) -> NetworkStatus {
        self.orchestrator.status().await
    }
}
