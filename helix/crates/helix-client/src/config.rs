//! HELIX Configuration Module
//!
//! Defines configuration structures for HELIX nodes, networks, and training.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Root configuration for a HELIX node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelixConfig {
    /// Node-specific configuration
    pub node: NodeConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Training configuration
    pub training: TrainingConfig,
}

impl Default for HelixConfig {
    fn default() -> Self {
        Self {
            node: NodeConfig::default(),
            network: NetworkConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

impl HelixConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Node-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Unique name for this node
    pub name: String,
    /// Network to connect to (mainnet, testnet, local)
    pub network: String,
    /// Directory for storing node data
    pub data_dir: PathBuf,
    /// Node capabilities (train, aggregate, prove)
    pub capabilities: Vec<String>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            name: format!("helix-node-{}", &uuid::Uuid::new_v4().to_string()[..8]),
            network: "local".to_string(),
            data_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".helix")
                .join("data"),
            capabilities: vec!["train".to_string(), "prove".to_string()],
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Address to listen on
    pub listen_addr: String,
    /// Bootstrap nodes for peer discovery
    pub bootstrap_nodes: Vec<String>,
    /// Enable mDNS for local peer discovery
    pub enable_mdns: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:9000".to_string(),
            bootstrap_nodes: vec![],
            enable_mdns: true,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size for training
    pub batch_size: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of training rounds
    pub max_rounds: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            max_rounds: 100,
        }
    }
}

/// Contract addresses configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractConfig {
    /// HelixCoordinator contract address
    pub coordinator: String,
    /// HelixVerifier contract address
    pub verifier: String,
    /// HelixToken contract address (optional)
    pub token: Option<String>,
}

impl Default for ContractConfig {
    fn default() -> Self {
        Self {
            coordinator: String::new(),
            verifier: String::new(),
            token: None,
        }
    }
}

/// RPC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcConfig {
    /// RPC endpoint URL
    pub url: String,
    /// Chain ID
    pub chain_id: u64,
    /// Request timeout in seconds
    pub timeout: u64,
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8545".to_string(),
            chain_id: 31337,
            timeout: 30,
        }
    }
}
