//! Node Configuration.
//!
//! Provides configuration for HELIX nodes including network settings,
//! storage paths, and role assignments. Supports loading from and saving
//! to JSON config files.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Role of this node in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeRole {
    /// Performs training computations and generates proofs.
    Compute,
    /// Aggregates gradients from compute nodes.
    Aggregator,
    /// Verifies proofs submitted by compute nodes.
    Verifier,
}

impl Default for NodeRole {
    fn default() -> Self {
        Self::Compute
    }
}

/// Rate limit configuration for the node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum messages per second per peer.
    pub max_messages_per_sec: u32,
    /// Maximum bytes per second per peer.
    pub max_bytes_per_sec: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_messages_per_sec: 100,
            max_bytes_per_sec: 10 * 1024 * 1024, // 10 MB/s
        }
    }
}

/// HTTP API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Bearer token for authenticating mutating API requests.
    /// If None, one is generated automatically on startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Maximum requests per second per IP address (token bucket).
    #[serde(default = "default_api_rate_limit")]
    pub rate_limit_per_sec: u32,

    /// Address to bind the HTTP API server.
    #[serde(default = "default_api_listen_addr")]
    pub listen_addr: String,
}

fn default_api_rate_limit() -> u32 {
    100
}

fn default_api_listen_addr() -> String {
    "0.0.0.0:8080".to_string()
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            rate_limit_per_sec: default_api_rate_limit(),
            listen_addr: default_api_listen_addr(),
        }
    }
}

impl ApiConfig {
    /// Returns the API key, generating one if not already set.
    pub fn api_key_or_generate(&mut self) -> String {
        if let Some(ref key) = self.api_key {
            key.clone()
        } else {
            use rand::Rng;
            let key: String = rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(48)
                .map(char::from)
                .collect();
            self.api_key = Some(key.clone());
            key
        }
    }
}

/// Worker training data source configuration.
///
/// Controls where workers load their training data from. Defaults to
/// synthetic data generation for development/testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataSourceConfig {
    /// Generate synthetic data from a seed (current default behavior).
    Synthetic {
        /// Base seed for data generation.
        #[serde(default = "default_data_seed")]
        seed: u64,
    },
    /// Load training data from a local CSV file.
    CsvFile {
        /// Path to the CSV file.
        path: String,
        /// Column names to use as input features.
        input_cols: Vec<String>,
        /// Column names to use as targets.
        target_cols: Vec<String>,
    },
    /// Load training data from IPFS.
    Ipfs {
        /// Content identifier for the dataset.
        cid: String,
        /// IPFS gateway URL.
        #[serde(default = "default_ipfs_gateway")]
        gateway: String,
    },
}

fn default_data_seed() -> u64 {
    42
}

fn default_ipfs_gateway() -> String {
    "http://localhost:5001".to_string()
}

impl Default for DataSourceConfig {
    fn default() -> Self {
        Self::Synthetic {
            seed: default_data_seed(),
        }
    }
}

/// MPC (Multi-Party Computation) configuration.
///
/// Controls whether MPC is used for privacy-preserving training.
/// MPC is enabled by default — the non-MPC path is opt-out for testing/debugging only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpcConfig {
    /// Whether MPC is enabled. Defaults to true (privacy-first).
    #[serde(default = "default_mpc_enabled")]
    pub enabled: bool,

    /// This worker's party index in the MPC protocol.
    /// Auto-assigned by the aggregator if not set.
    #[serde(default)]
    pub party_index: Option<usize>,

    /// Total number of parties in the MPC protocol.
    #[serde(default = "default_mpc_num_parties")]
    pub num_parties: usize,

    /// How often to reshare weights (in steps) to prevent gradient accumulation attacks.
    #[serde(default = "default_mpc_reshare_interval")]
    pub reshare_interval: u64,

    /// Maximum gradient norm for anomaly detection.
    #[serde(default = "default_mpc_max_gradient_norm")]
    pub max_gradient_norm: f64,

    /// Whether to verify share commitments.
    #[serde(default = "default_mpc_verify_commitments")]
    pub verify_commitments: bool,

    /// Password for encrypted weight storage (used with AES-256-GCM + Argon2id).
    /// Read from HELIX_MPC_STORAGE_KEY env var. Never stored in config files.
    #[serde(skip)]
    pub storage_key: Option<String>,
}

fn default_mpc_enabled() -> bool {
    true
}

fn default_mpc_num_parties() -> usize {
    3
}

fn default_mpc_reshare_interval() -> u64 {
    50
}

fn default_mpc_max_gradient_norm() -> f64 {
    100.0
}

fn default_mpc_verify_commitments() -> bool {
    true
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            enabled: default_mpc_enabled(),
            party_index: None,
            num_parties: default_mpc_num_parties(),
            reshare_interval: default_mpc_reshare_interval(),
            max_gradient_norm: default_mpc_max_gradient_norm(),
            verify_commitments: default_mpc_verify_commitments(),
            storage_key: None,
        }
    }
}

impl MpcConfig {
    /// Returns the storage key from the HELIX_MPC_STORAGE_KEY environment variable.
    pub fn storage_key(&self) -> Option<String> {
        self.storage_key.clone().or_else(|| std::env::var("HELIX_MPC_STORAGE_KEY").ok())
    }
}

/// Fault tolerance configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceNodeConfig {
    /// How often to checkpoint worker state (in training steps).
    /// 0 = only checkpoint on round completion.
    #[serde(default = "default_checkpoint_interval_steps")]
    pub checkpoint_interval_steps: u64,

    /// Heartbeat timeout before marking a worker as suspected (seconds).
    #[serde(default = "default_heartbeat_timeout_secs")]
    pub heartbeat_timeout_secs: u64,

    /// Maximum missed heartbeats before marking worker as failed.
    #[serde(default = "default_max_missed_heartbeats")]
    pub max_missed_heartbeats: u32,

    /// Minimum healthy workers required to continue a round.
    #[serde(default = "default_min_healthy_workers")]
    pub min_healthy_workers: usize,

    /// Maximum failures before a worker is excluded from future rounds.
    #[serde(default = "default_max_failures_before_exclusion")]
    pub max_failures_before_exclusion: u32,

    /// Cooldown period after failure before a worker can rejoin (seconds).
    #[serde(default = "default_failure_cooldown_secs")]
    pub failure_cooldown_secs: u64,

    /// Enable automatic worker recovery.
    #[serde(default = "default_auto_recovery")]
    pub auto_recovery: bool,

    /// Maximum number of checkpoints to keep on disk.
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: usize,

    /// Graceful shutdown timeout (seconds). After this, force exit.
    #[serde(default = "default_shutdown_timeout_secs")]
    pub shutdown_timeout_secs: u64,
}

fn default_checkpoint_interval_steps() -> u64 { 10 }
fn default_heartbeat_timeout_secs() -> u64 { 15 }
fn default_max_missed_heartbeats() -> u32 { 3 }
fn default_min_healthy_workers() -> usize { 1 }
fn default_max_failures_before_exclusion() -> u32 { 3 }
fn default_failure_cooldown_secs() -> u64 { 60 }
fn default_auto_recovery() -> bool { true }
fn default_max_checkpoints() -> usize { 5 }
fn default_shutdown_timeout_secs() -> u64 { 30 }

impl Default for FaultToleranceNodeConfig {
    fn default() -> Self {
        Self {
            checkpoint_interval_steps: default_checkpoint_interval_steps(),
            heartbeat_timeout_secs: default_heartbeat_timeout_secs(),
            max_missed_heartbeats: default_max_missed_heartbeats(),
            min_healthy_workers: default_min_healthy_workers(),
            max_failures_before_exclusion: default_max_failures_before_exclusion(),
            failure_cooldown_secs: default_failure_cooldown_secs(),
            auto_recovery: default_auto_recovery(),
            max_checkpoints: default_max_checkpoints(),
            shutdown_timeout_secs: default_shutdown_timeout_secs(),
        }
    }
}

/// Full node configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Address to listen on for P2P connections.
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,

    /// Directory for persistent data (state, checkpoints, peers).
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Role of this node.
    #[serde(default)]
    pub role: NodeRole,

    /// Ethereum RPC URL for on-chain interaction.
    #[serde(default = "default_rpc_url")]
    pub rpc_url: String,

    /// Private key for signing transactions (hex, without 0x prefix).
    /// SECURITY: Never stored in config files. Read from HELIX_PRIVATE_KEY env var.
    #[serde(skip)]
    pub private_key: Option<String>,

    /// Whether to use TLS for peer connections.
    #[serde(default = "default_use_tls")]
    pub use_tls: bool,

    /// Gossip send interval in milliseconds.
    #[serde(default = "default_gossip_interval")]
    pub gossip_send_interval_ms: u64,

    /// Maximum outbound messages per tick.
    #[serde(default = "default_max_outbound")]
    pub max_outbound_per_tick: usize,

    /// Rate limiting configuration.
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// HTTP API configuration.
    #[serde(default)]
    pub api: ApiConfig,

    /// MPC (Multi-Party Computation) configuration.
    /// Enabled by default for privacy-preserving training.
    #[serde(default)]
    pub mpc: MpcConfig,

    /// On-chain pipeline configuration for aggregator nodes.
    /// When set, the aggregator will register models, stake, submit proofs on-chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<ChainConfig>,

    /// Training data source configuration for worker nodes.
    #[serde(default)]
    pub data_source: DataSourceConfig,

    /// Directory to persist model checkpoints (aggregator only).
    #[serde(default = "default_checkpoint_dir")]
    pub checkpoint_dir: PathBuf,

    /// Fault tolerance and crash recovery configuration.
    #[serde(default)]
    pub fault_tolerance: FaultToleranceNodeConfig,
}

/// On-chain pipeline configuration for aggregator nodes.
///
/// When an aggregator starts with this config, it will:
/// 1. Register the model on-chain (if `model_id` is None)
/// 2. Stake tokens for proof submission rights
/// 3. Start training rounds on-chain
/// 4. Submit aggregated proofs via the coordinator contract
/// 5. Use VerifyAll verification with real Halo2 KZG checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    /// Deployed HelixCoordinatorV2 contract address (hex, with 0x prefix).
    pub coordinator_address: String,

    /// IPFS hash for the model metadata.
    #[serde(default = "default_model_ipfs_hash")]
    pub model_ipfs_hash: String,

    /// Pre-registered model ID. If None, a new model is registered on startup.
    #[serde(default)]
    pub model_id: Option<u64>,

    /// Minimum stake required for the model (in wei).
    #[serde(default = "default_min_stake_wei")]
    pub min_stake_wei: String,

    /// Amount to stake (in wei).
    #[serde(default = "default_stake_amount_wei")]
    pub stake_amount_wei: String,

    /// Round duration in seconds.
    #[serde(default = "default_round_duration_secs")]
    pub round_duration_secs: u64,

    /// Model input dimension.
    pub d_in: usize,

    /// Model hidden dimension.
    pub d_hid: usize,

    /// Model output dimension.
    pub d_out: usize,

    /// Interval in seconds for processing the proof queue.
    #[serde(default = "default_proof_queue_interval_secs")]
    pub proof_queue_interval_secs: u64,
}

fn default_model_ipfs_hash() -> String {
    "QmDefault".to_string()
}

fn default_min_stake_wei() -> String {
    "1000000000000000000".to_string() // 1 ETH
}

fn default_stake_amount_wei() -> String {
    "1000000000000000000".to_string() // 1 ETH
}

fn default_round_duration_secs() -> u64 {
    600 // 10 minutes
}

fn default_proof_queue_interval_secs() -> u64 {
    5
}

fn default_checkpoint_dir() -> PathBuf {
    dirs_fallback().join("checkpoints")
}

fn default_listen_addr() -> String {
    "0.0.0.0:9000".to_string()
}

fn default_data_dir() -> PathBuf {
    dirs_fallback()
}

fn default_rpc_url() -> String {
    "http://localhost:8545".to_string()
}

fn default_use_tls() -> bool {
    true
}

fn default_gossip_interval() -> u64 {
    100
}

fn default_max_outbound() -> usize {
    50
}

/// Fallback for data directory: ~/.helix/ or /tmp/helix/
fn dirs_fallback() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".helix")
    } else {
        PathBuf::from("/tmp/helix")
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            listen_addr: default_listen_addr(),
            data_dir: default_data_dir(),
            role: NodeRole::default(),
            rpc_url: default_rpc_url(),
            private_key: None,
            use_tls: default_use_tls(),
            gossip_send_interval_ms: default_gossip_interval(),
            max_outbound_per_tick: default_max_outbound(),
            rate_limit: RateLimitConfig::default(),
            api: ApiConfig::default(),
            mpc: MpcConfig::default(),
            chain: None,
            data_source: DataSourceConfig::default(),
            checkpoint_dir: default_checkpoint_dir(),
            fault_tolerance: FaultToleranceNodeConfig::default(),
        }
    }
}

impl NodeConfig {
    /// Returns the private key from the HELIX_PRIVATE_KEY environment variable.
    ///
    /// Private keys are never stored in config files for security.
    /// Set the `HELIX_PRIVATE_KEY` env var (hex, without 0x prefix) before starting the node.
    pub fn private_key(&self) -> Option<String> {
        self.private_key.clone().or_else(|| std::env::var("HELIX_PRIVATE_KEY").ok())
    }

    /// Loads configuration from a JSON file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let data = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::Io(e.to_string()))?;

        // Warn if the config file contains a private_key field
        if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&data) {
            if raw.get("private_key").is_some() {
                log::warn!(
                    "Config file contains 'private_key' field — this is ignored for security. \
                     Use the HELIX_PRIVATE_KEY environment variable instead."
                );
            }
        }

        let config: Self = serde_json::from_str(&data)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Saves configuration to a JSON file.
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        std::fs::write(path.as_ref(), data)
            .map_err(|e| ConfigError::Io(e.to_string()))?;
        Ok(())
    }

    /// Validates the configuration for consistency.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.listen_addr.is_empty() {
            return Err(ConfigError::Validation("listen_addr cannot be empty".into()));
        }

        // Validate listen_addr is a valid socket address
        if self.listen_addr.parse::<std::net::SocketAddr>().is_err() {
            return Err(ConfigError::Validation(format!(
                "listen_addr '{}' is not a valid socket address",
                self.listen_addr
            )));
        }

        if self.rpc_url.is_empty() {
            return Err(ConfigError::Validation("rpc_url cannot be empty".into()));
        }

        if self.rate_limit.max_messages_per_sec == 0 {
            return Err(ConfigError::Validation("max_messages_per_sec must be > 0".into()));
        }

        Ok(())
    }
}

/// Configuration errors.
#[derive(Debug)]
pub enum ConfigError {
    /// IO error reading/writing config file.
    Io(String),
    /// Parse error in config file.
    Parse(String),
    /// Validation error.
    Validation(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "Config IO error: {}", msg),
            Self::Parse(msg) => write!(f, "Config parse error: {}", msg),
            Self::Validation(msg) => write!(f, "Config validation error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = NodeConfig::default();
        config.validate().unwrap();
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = NodeConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let loaded: NodeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.listen_addr, config.listen_addr);
        assert_eq!(loaded.role, config.role);
        assert_eq!(loaded.use_tls, config.use_tls);
    }

    #[test]
    fn test_config_from_file_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");

        let config = NodeConfig {
            role: NodeRole::Aggregator,
            listen_addr: "127.0.0.1:9001".to_string(),
            ..Default::default()
        };

        config.to_file(&path).unwrap();
        let loaded = NodeConfig::from_file(&path).unwrap();

        assert_eq!(loaded.role, NodeRole::Aggregator);
        assert_eq!(loaded.listen_addr, "127.0.0.1:9001");
    }

    #[test]
    fn test_config_validation_empty_listen_addr() {
        let config = NodeConfig {
            listen_addr: "".to_string(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_listen_addr() {
        let config = NodeConfig {
            listen_addr: "not-a-socket-addr".to_string(),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_rate_limit() {
        let config = NodeConfig {
            rate_limit: RateLimitConfig {
                max_messages_per_sec: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_private_key_never_serialized() {
        // With #[serde(skip)], private_key is never serialized even if set
        let mut config = NodeConfig::default();
        config.private_key = Some("secret_hex_key".to_string());
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("private_key"), "private_key must never appear in serialized config");
        assert!(!json.contains("secret_hex_key"), "private key value must never appear in JSON");
    }

    #[test]
    fn test_private_key_from_env() {
        let config = NodeConfig::default();
        // When env var is not set and field is None, should return None
        std::env::remove_var("HELIX_PRIVATE_KEY");
        assert!(config.private_key().is_none());

        // When env var is set, should return it
        std::env::set_var("HELIX_PRIVATE_KEY", "deadbeef");
        assert_eq!(config.private_key(), Some("deadbeef".to_string()));
        std::env::remove_var("HELIX_PRIVATE_KEY");
    }
}
