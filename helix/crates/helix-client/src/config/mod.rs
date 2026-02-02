//! HELIX Configuration Module
//!
//! Defines configuration structures for HELIX nodes, networks, and training.
//! Includes pre-defined profiles for different deployment scenarios:
//! - Local: Local development with Anvil/Hardhat
//! - Anvil: Anvil fork for testing
//! - Sepolia: Sepolia testnet deployment
//! - Mainnet: Ethereum mainnet production
//! - Custom: User-defined configuration

pub mod training;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub use training::TrainingJobConfig;

// ============================================================================
// Root Configuration
// ============================================================================

/// Root configuration for a HELIX node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelixConfig {
    /// Configuration profile used
    #[serde(default)]
    pub profile: ConfigProfile,
    /// Node-specific configuration
    pub node: NodeConfig,
    /// Network configuration
    pub network: NetworkConfig,
    /// Training configuration
    pub training: TrainingConfig,
    /// RPC configuration
    #[serde(default)]
    pub rpc: RpcConfig,
    /// Contract addresses
    #[serde(default)]
    pub contracts: ContractConfig,
    /// Staking configuration
    #[serde(default)]
    pub staking: StakingConfig,
    /// Proof configuration
    #[serde(default)]
    pub proof: ProofConfig,
    /// Storage configuration
    #[serde(default)]
    pub storage: StorageConfig,
    /// Telemetry configuration
    #[serde(default)]
    pub telemetry: TelemetryConfig,
}

impl Default for HelixConfig {
    fn default() -> Self {
        Self {
            profile: ConfigProfile::Local,
            node: NodeConfig::default(),
            network: NetworkConfig::default(),
            training: TrainingConfig::default(),
            rpc: RpcConfig::default(),
            contracts: ContractConfig::default(),
            staking: StakingConfig::default(),
            proof: ProofConfig::default(),
            storage: StorageConfig::default(),
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl HelixConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create configuration from a profile
    pub fn from_profile(profile: ConfigProfile) -> Self {
        profile.to_config()
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge(&mut self, other: &HelixConfig) {
        // Only merge non-default values
        if other.profile != ConfigProfile::Local {
            self.profile = other.profile.clone();
        }
        // Node config
        if !other.node.name.is_empty() {
            self.node.name = other.node.name.clone();
        }
        if !other.node.network.is_empty() {
            self.node.network = other.node.network.clone();
        }
        // RPC
        if !other.rpc.url.is_empty() && other.rpc.url != "http://localhost:8545" {
            self.rpc = other.rpc.clone();
        }
        // Contracts
        if !other.contracts.coordinator.is_empty() {
            self.contracts = other.contracts.clone();
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate node name
        if self.node.name.is_empty() {
            return Err(anyhow!("Node name cannot be empty"));
        }

        // Validate RPC URL
        if self.rpc.url.is_empty() {
            return Err(anyhow!("RPC URL cannot be empty"));
        }

        // Validate training config
        if self.training.batch_size == 0 {
            return Err(anyhow!("Batch size must be > 0"));
        }
        if self.training.learning_rate <= 0.0 {
            return Err(anyhow!("Learning rate must be > 0"));
        }
        if self.training.max_rounds == 0 {
            return Err(anyhow!("Max rounds must be > 0"));
        }

        // Validate staking config
        if self.staking.min_stake < 0.0 {
            return Err(anyhow!("Min stake cannot be negative"));
        }

        // Validate proof config
        if self.proof.error_bound_max <= 0.0 {
            return Err(anyhow!("Error bound max must be > 0"));
        }

        Ok(())
    }

    /// Generate example configuration files for all profiles
    pub fn generate_examples(output_dir: &PathBuf) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        for profile in ConfigProfile::all() {
            let config = profile.to_config();
            let filename = format!("helix-{}.toml", profile.name().to_lowercase().replace(' ', "-"));
            let path = output_dir.join(filename);
            config.save(&path)?;
        }

        Ok(())
    }
}

// ============================================================================
// Configuration Profiles
// ============================================================================

/// Pre-defined configuration profiles for different deployment scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigProfile {
    /// Local development (Anvil/Hardhat)
    Local,
    /// Anvil fork for testing
    Anvil,
    /// Sepolia testnet
    Sepolia,
    /// Ethereum mainnet
    Mainnet,
    /// Custom configuration
    Custom,
}

impl Default for ConfigProfile {
    fn default() -> Self {
        Self::Local
    }
}

impl ConfigProfile {
    /// Get all available profiles
    pub fn all() -> Vec<Self> {
        vec![Self::Local, Self::Anvil, Self::Sepolia, Self::Mainnet, Self::Custom]
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Local => "Local Development",
            Self::Anvil => "Anvil Testnet",
            Self::Sepolia => "Sepolia Testnet",
            Self::Mainnet => "Ethereum Mainnet",
            Self::Custom => "Custom",
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Local => "Local development with Anvil/Hardhat node, fast block times, unlimited ETH",
            Self::Anvil => "Anvil instance forking mainnet state for realistic testing",
            Self::Sepolia => "Sepolia testnet for pre-production testing with real network conditions",
            Self::Mainnet => "Ethereum mainnet for production deployment (use with caution)",
            Self::Custom => "Custom configuration for specialized deployments",
        }
    }

    /// Convert profile to full configuration
    pub fn to_config(&self) -> HelixConfig {
        match self {
            Self::Local => self.local_config(),
            Self::Anvil => self.anvil_config(),
            Self::Sepolia => self.sepolia_config(),
            Self::Mainnet => self.mainnet_config(),
            Self::Custom => HelixConfig::default(),
        }
    }

    fn local_config(&self) -> HelixConfig {
        HelixConfig {
            profile: Self::Local,
            node: NodeConfig {
                name: format!("helix-local-{}", &uuid::Uuid::new_v4().to_string()[..8]),
                network: "local".to_string(),
                data_dir: dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".helix")
                    .join("local"),
                capabilities: vec!["train".to_string(), "aggregate".to_string(), "prove".to_string()],
                role: NodeRole::Worker,
                max_concurrent_models: 3,
            },
            network: NetworkConfig {
                listen_addr: "0.0.0.0:9000".to_string(),
                bootstrap_nodes: vec![],
                enable_mdns: true,
                max_peers: 50,
                connection_timeout: 30,
                heartbeat_interval: 10,
            },
            training: TrainingConfig {
                batch_size: 32,
                learning_rate: 0.001,
                max_rounds: 100,
                round_timeout: 120,
                gradient_compression: false,
                checkpoint_interval: 10,
            },
            rpc: RpcConfig {
                url: "http://localhost:8545".to_string(),
                chain_id: 31337,
                timeout: 30,
                max_retries: 3,
                retry_delay: 1,
            },
            contracts: ContractConfig {
                coordinator: "0x5FbDB2315678afecb367f032d93F642f64180aa3".to_string(),
                verifier: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512".to_string(),
                staking: Some("0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0".to_string()),
                token: None,
                approximate_verifier: Some("0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9".to_string()),
            },
            staking: StakingConfig {
                min_stake: 0.1,
                default_stake: 0.5,
                lock_period: 0, // No lock for local testing
                auto_stake: true,
                slashing_enabled: true,
            },
            proof: ProofConfig {
                proof_system: ProofSystem::Approximate,
                error_bound_max: 1000.0,
                verification_timeout: 60,
                batch_proofs: true,
                proof_compression: false,
            },
            storage: StorageConfig {
                ipfs_gateway: "http://localhost:5001".to_string(),
                model_cache_size: 1024, // MB
                checkpoint_retention: 10,
                enable_pinning: false,
            },
            telemetry: TelemetryConfig {
                enabled: true,
                endpoint: None,
                metrics_interval: 10,
                log_level: "debug".to_string(),
                trace_sampling: 1.0,
            },
        }
    }

    fn anvil_config(&self) -> HelixConfig {
        let mut config = self.local_config();
        config.profile = Self::Anvil;
        config.node.name = format!("helix-anvil-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        config.node.network = "anvil".to_string();
        config.node.data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".helix")
            .join("anvil");
        config.rpc.url = "http://localhost:8545".to_string();
        config.rpc.chain_id = 31337;
        config.staking.lock_period = 3600; // 1 hour for testing
        config.telemetry.log_level = "info".to_string();
        config
    }

    fn sepolia_config(&self) -> HelixConfig {
        HelixConfig {
            profile: Self::Sepolia,
            node: NodeConfig {
                name: format!("helix-sepolia-{}", &uuid::Uuid::new_v4().to_string()[..8]),
                network: "sepolia".to_string(),
                data_dir: dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".helix")
                    .join("sepolia"),
                capabilities: vec!["train".to_string(), "prove".to_string()],
                role: NodeRole::Worker,
                max_concurrent_models: 1,
            },
            network: NetworkConfig {
                listen_addr: "0.0.0.0:9000".to_string(),
                bootstrap_nodes: vec![
                    "/dns4/helix-bootstrap-sepolia-1.example.com/tcp/9000/p2p/QmBootstrap1...".to_string(),
                    "/dns4/helix-bootstrap-sepolia-2.example.com/tcp/9000/p2p/QmBootstrap2...".to_string(),
                ],
                enable_mdns: false,
                max_peers: 100,
                connection_timeout: 60,
                heartbeat_interval: 30,
            },
            training: TrainingConfig {
                batch_size: 32,
                learning_rate: 0.001,
                max_rounds: 100,
                round_timeout: 300,
                gradient_compression: true,
                checkpoint_interval: 5,
            },
            rpc: RpcConfig {
                url: "https://sepolia.infura.io/v3/YOUR_INFURA_KEY".to_string(),
                chain_id: 11155111,
                timeout: 60,
                max_retries: 5,
                retry_delay: 2,
            },
            contracts: ContractConfig {
                coordinator: "".to_string(), // To be filled after deployment
                verifier: "".to_string(),
                staking: None,
                token: None,
                approximate_verifier: None,
            },
            staking: StakingConfig {
                min_stake: 0.01,
                default_stake: 0.1,
                lock_period: 86400, // 1 day
                auto_stake: false,
                slashing_enabled: true,
            },
            proof: ProofConfig {
                proof_system: ProofSystem::Approximate,
                error_bound_max: 1000.0,
                verification_timeout: 120,
                batch_proofs: true,
                proof_compression: true,
            },
            storage: StorageConfig {
                ipfs_gateway: "https://ipfs.infura.io:5001".to_string(),
                model_cache_size: 512,
                checkpoint_retention: 5,
                enable_pinning: true,
            },
            telemetry: TelemetryConfig {
                enabled: true,
                endpoint: Some("https://telemetry.helix.example.com".to_string()),
                metrics_interval: 30,
                log_level: "info".to_string(),
                trace_sampling: 0.1,
            },
        }
    }

    fn mainnet_config(&self) -> HelixConfig {
        HelixConfig {
            profile: Self::Mainnet,
            node: NodeConfig {
                name: format!("helix-mainnet-{}", &uuid::Uuid::new_v4().to_string()[..8]),
                network: "mainnet".to_string(),
                data_dir: dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".helix")
                    .join("mainnet"),
                capabilities: vec!["train".to_string(), "prove".to_string()],
                role: NodeRole::Worker,
                max_concurrent_models: 1,
            },
            network: NetworkConfig {
                listen_addr: "0.0.0.0:9000".to_string(),
                bootstrap_nodes: vec![
                    "/dns4/helix-bootstrap-mainnet-1.helix.network/tcp/9000/p2p/QmMainnet1...".to_string(),
                    "/dns4/helix-bootstrap-mainnet-2.helix.network/tcp/9000/p2p/QmMainnet2...".to_string(),
                    "/dns4/helix-bootstrap-mainnet-3.helix.network/tcp/9000/p2p/QmMainnet3...".to_string(),
                ],
                enable_mdns: false,
                max_peers: 200,
                connection_timeout: 90,
                heartbeat_interval: 60,
            },
            training: TrainingConfig {
                batch_size: 64,
                learning_rate: 0.0001,
                max_rounds: 1000,
                round_timeout: 600,
                gradient_compression: true,
                checkpoint_interval: 1,
            },
            rpc: RpcConfig {
                url: "https://mainnet.infura.io/v3/YOUR_INFURA_KEY".to_string(),
                chain_id: 1,
                timeout: 120,
                max_retries: 10,
                retry_delay: 5,
            },
            contracts: ContractConfig {
                coordinator: "".to_string(), // To be filled after deployment
                verifier: "".to_string(),
                staking: None,
                token: None,
                approximate_verifier: None,
            },
            staking: StakingConfig {
                min_stake: 0.1,
                default_stake: 1.0,
                lock_period: 604800, // 7 days
                auto_stake: false,
                slashing_enabled: true,
            },
            proof: ProofConfig {
                proof_system: ProofSystem::Approximate,
                error_bound_max: 1000.0,
                verification_timeout: 300,
                batch_proofs: true,
                proof_compression: true,
            },
            storage: StorageConfig {
                ipfs_gateway: "https://ipfs.infura.io:5001".to_string(),
                model_cache_size: 2048,
                checkpoint_retention: 10,
                enable_pinning: true,
            },
            telemetry: TelemetryConfig {
                enabled: true,
                endpoint: Some("https://telemetry.helix.network".to_string()),
                metrics_interval: 60,
                log_level: "warn".to_string(),
                trace_sampling: 0.01,
            },
        }
    }
}

impl std::fmt::Display for ConfigProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for ConfigProfile {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "local" | "dev" | "development" => Ok(Self::Local),
            "anvil" | "fork" => Ok(Self::Anvil),
            "sepolia" | "testnet" => Ok(Self::Sepolia),
            "mainnet" | "production" | "prod" => Ok(Self::Mainnet),
            "custom" => Ok(Self::Custom),
            _ => Err(anyhow!("Unknown profile: {}. Valid options: local, anvil, sepolia, mainnet, custom", s)),
        }
    }
}

// ============================================================================
// Node Configuration
// ============================================================================

/// Node role in the network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Training worker
    Worker,
    /// Gradient aggregator
    Aggregator,
    /// Network coordinator (for demo/testing)
    Coordinator,
}

impl Default for NodeRole {
    fn default() -> Self {
        Self::Worker
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
    /// Node role
    #[serde(default)]
    pub role: NodeRole,
    /// Maximum concurrent models to train
    #[serde(default = "default_max_concurrent_models")]
    pub max_concurrent_models: u32,
}

fn default_max_concurrent_models() -> u32 {
    1
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
            role: NodeRole::Worker,
            max_concurrent_models: 1,
        }
    }
}

// ============================================================================
// Network Configuration
// ============================================================================

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Address to listen on
    pub listen_addr: String,
    /// Bootstrap nodes for peer discovery
    pub bootstrap_nodes: Vec<String>,
    /// Enable mDNS for local peer discovery
    pub enable_mdns: bool,
    /// Maximum number of peers
    #[serde(default = "default_max_peers")]
    pub max_peers: u32,
    /// Connection timeout in seconds
    #[serde(default = "default_connection_timeout")]
    pub connection_timeout: u64,
    /// Heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval: u64,
}

fn default_max_peers() -> u32 {
    50
}

fn default_connection_timeout() -> u64 {
    30
}

fn default_heartbeat_interval() -> u64 {
    10
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:9000".to_string(),
            bootstrap_nodes: vec![],
            enable_mdns: true,
            max_peers: 50,
            connection_timeout: 30,
            heartbeat_interval: 10,
        }
    }
}

// ============================================================================
// Training Configuration
// ============================================================================

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Batch size for training
    pub batch_size: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of training rounds
    pub max_rounds: u32,
    /// Round timeout in seconds
    #[serde(default = "default_round_timeout")]
    pub round_timeout: u64,
    /// Enable gradient compression
    #[serde(default)]
    pub gradient_compression: bool,
    /// Checkpoint interval (rounds)
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: u32,
}

fn default_round_timeout() -> u64 {
    120
}

fn default_checkpoint_interval() -> u32 {
    10
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
            max_rounds: 100,
            round_timeout: 120,
            gradient_compression: false,
            checkpoint_interval: 10,
        }
    }
}

// ============================================================================
// Contract Configuration
// ============================================================================

/// Contract addresses configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractConfig {
    /// HelixCoordinator contract address
    pub coordinator: String,
    /// HelixVerifier contract address
    pub verifier: String,
    /// HelixStaking contract address
    pub staking: Option<String>,
    /// HelixToken contract address (optional)
    pub token: Option<String>,
    /// ApproximateProofVerifier contract address
    pub approximate_verifier: Option<String>,
}

impl Default for ContractConfig {
    fn default() -> Self {
        Self {
            coordinator: String::new(),
            verifier: String::new(),
            staking: None,
            token: None,
            approximate_verifier: None,
        }
    }
}

impl ContractConfig {
    /// Check if all required contracts are configured
    pub fn is_configured(&self) -> bool {
        !self.coordinator.is_empty() && !self.verifier.is_empty()
    }

    /// Validate contract addresses
    pub fn validate(&self) -> Result<()> {
        if !self.coordinator.is_empty() && !self.coordinator.starts_with("0x") {
            return Err(anyhow!("Invalid coordinator address format"));
        }
        if !self.verifier.is_empty() && !self.verifier.starts_with("0x") {
            return Err(anyhow!("Invalid verifier address format"));
        }
        Ok(())
    }
}

// ============================================================================
// RPC Configuration
// ============================================================================

/// RPC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcConfig {
    /// RPC endpoint URL
    pub url: String,
    /// Chain ID
    pub chain_id: u64,
    /// Request timeout in seconds
    pub timeout: u64,
    /// Maximum retries for failed requests
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Delay between retries in seconds
    #[serde(default = "default_retry_delay")]
    pub retry_delay: u64,
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_delay() -> u64 {
    1
}

impl Default for RpcConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:8545".to_string(),
            chain_id: 31337,
            timeout: 30,
            max_retries: 3,
            retry_delay: 1,
        }
    }
}

// ============================================================================
// Staking Configuration
// ============================================================================

/// Staking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingConfig {
    /// Minimum stake required (ETH)
    pub min_stake: f64,
    /// Default stake amount (ETH)
    pub default_stake: f64,
    /// Lock period in seconds
    pub lock_period: u64,
    /// Automatically stake when joining
    #[serde(default)]
    pub auto_stake: bool,
    /// Enable slashing for misbehavior
    #[serde(default = "default_slashing_enabled")]
    pub slashing_enabled: bool,
}

fn default_slashing_enabled() -> bool {
    true
}

impl Default for StakingConfig {
    fn default() -> Self {
        Self {
            min_stake: 0.1,
            default_stake: 0.5,
            lock_period: 604800, // 7 days
            auto_stake: false,
            slashing_enabled: true,
        }
    }
}

// ============================================================================
// Proof Configuration
// ============================================================================

/// Proof system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofSystem {
    /// Approximate ZK proofs (faster, bounded error)
    Approximate,
    /// Full ZK proofs (slower, exact)
    Full,
    /// Optimistic proofs with fraud detection
    Optimistic,
}

impl Default for ProofSystem {
    fn default() -> Self {
        Self::Approximate
    }
}

/// Proof configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofConfig {
    /// Proof system to use
    #[serde(default)]
    pub proof_system: ProofSystem,
    /// Maximum allowed error bound
    pub error_bound_max: f64,
    /// Verification timeout in seconds
    pub verification_timeout: u64,
    /// Batch multiple proofs together
    #[serde(default)]
    pub batch_proofs: bool,
    /// Compress proofs for storage
    #[serde(default)]
    pub proof_compression: bool,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            proof_system: ProofSystem::Approximate,
            error_bound_max: 1000.0,
            verification_timeout: 60,
            batch_proofs: true,
            proof_compression: false,
        }
    }
}

// ============================================================================
// Storage Configuration
// ============================================================================

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// IPFS gateway URL
    pub ipfs_gateway: String,
    /// Model cache size in MB
    pub model_cache_size: u64,
    /// Number of checkpoints to retain
    pub checkpoint_retention: u32,
    /// Enable IPFS pinning for important data
    #[serde(default)]
    pub enable_pinning: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            ipfs_gateway: "http://localhost:5001".to_string(),
            model_cache_size: 1024,
            checkpoint_retention: 10,
            enable_pinning: false,
        }
    }
}

// ============================================================================
// Telemetry Configuration
// ============================================================================

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable telemetry
    pub enabled: bool,
    /// Telemetry endpoint URL
    pub endpoint: Option<String>,
    /// Metrics collection interval in seconds
    pub metrics_interval: u64,
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    /// Trace sampling rate (0.0 - 1.0)
    #[serde(default = "default_trace_sampling")]
    pub trace_sampling: f64,
}

fn default_trace_sampling() -> f64 {
    0.1
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: None,
            metrics_interval: 10,
            log_level: "info".to_string(),
            trace_sampling: 0.1,
        }
    }
}

// ============================================================================
// Profile Manager
// ============================================================================

/// Manages configuration profiles and switching between them
pub struct ProfileManager {
    /// Current active profile
    current_profile: ConfigProfile,
    /// Cached configurations by profile
    configs: HashMap<ConfigProfile, HelixConfig>,
    /// Base configuration directory
    config_dir: PathBuf,
}

impl ProfileManager {
    /// Create a new profile manager
    pub fn new() -> Self {
        let config_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".helix")
            .join("profiles");

        Self {
            current_profile: ConfigProfile::Local,
            configs: HashMap::new(),
            config_dir,
        }
    }

    /// Get the current profile
    pub fn current(&self) -> ConfigProfile {
        self.current_profile
    }

    /// Switch to a different profile
    pub fn switch(&mut self, profile: ConfigProfile) -> Result<&HelixConfig> {
        self.current_profile = profile;
        self.get_config(profile)
    }

    /// Get configuration for a profile
    pub fn get_config(&mut self, profile: ConfigProfile) -> Result<&HelixConfig> {
        if !self.configs.contains_key(&profile) {
            // Try to load from disk first
            let path = self.config_dir.join(format!("{}.toml", profile.name().to_lowercase().replace(' ', "-")));
            let config = if path.exists() {
                HelixConfig::load(&path)?
            } else {
                profile.to_config()
            };
            self.configs.insert(profile, config);
        }

        Ok(self.configs.get(&profile).unwrap())
    }

    /// Save current configuration to disk
    pub fn save_current(&self) -> Result<()> {
        if let Some(config) = self.configs.get(&self.current_profile) {
            std::fs::create_dir_all(&self.config_dir)?;
            let path = self.config_dir.join(format!("{}.toml", self.current_profile.name().to_lowercase().replace(' ', "-")));
            config.save(&path)?;
        }
        Ok(())
    }

    /// List all available profiles
    pub fn list_profiles(&self) -> Vec<(ConfigProfile, bool)> {
        ConfigProfile::all()
            .into_iter()
            .map(|p| {
                let path = self.config_dir.join(format!("{}.toml", p.name().to_lowercase().replace(' ', "-")));
                (p, path.exists())
            })
            .collect()
    }

    /// Initialize all profile configuration files
    pub fn init_all(&self) -> Result<()> {
        HelixConfig::generate_examples(&self.config_dir)
    }
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HelixConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_profile_to_config() {
        for profile in ConfigProfile::all() {
            let config = profile.to_config();
            assert_eq!(config.profile, profile);
            assert!(config.validate().is_ok(), "Profile {:?} failed validation", profile);
        }
    }

    #[test]
    fn test_profile_parsing() {
        assert_eq!("local".parse::<ConfigProfile>().unwrap(), ConfigProfile::Local);
        assert_eq!("sepolia".parse::<ConfigProfile>().unwrap(), ConfigProfile::Sepolia);
        assert_eq!("mainnet".parse::<ConfigProfile>().unwrap(), ConfigProfile::Mainnet);
        assert!("invalid".parse::<ConfigProfile>().is_err());
    }

    #[test]
    fn test_config_validation() {
        let mut config = HelixConfig::default();
        assert!(config.validate().is_ok());

        config.training.batch_size = 0;
        assert!(config.validate().is_err());

        config.training.batch_size = 32;
        config.training.learning_rate = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_contract_validation() {
        let mut contracts = ContractConfig::default();
        assert!(contracts.validate().is_ok());

        contracts.coordinator = "0x1234".to_string();
        assert!(contracts.validate().is_ok());

        contracts.coordinator = "invalid".to_string();
        assert!(contracts.validate().is_err());
    }
}
