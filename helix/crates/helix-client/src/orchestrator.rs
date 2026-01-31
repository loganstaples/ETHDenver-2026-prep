//! HELIX Network Orchestrator
//!
//! Manages multi-node networks for demos and testing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Child;
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

        let handle = NodeHandle {
            id: node_id.to_string(),
            role,
            status: NodeStatus::Running,
            port,
            process: None, // In demo mode, we simulate nodes
        };

        let mut nodes = self.nodes.write().await;
        nodes.insert(node_id.to_string(), handle);

        Ok(())
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
                nodes.remove(&id);
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
