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
