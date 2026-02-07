//! HelixClient SDK Facade
//!
//! Provides a high-level API for interacting with the HELIX protocol.
//! Wraps configuration, RPC communication, and optional dashboard state
//! into a single ergonomic entry point.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use crate::config::{ConfigProfile, HelixConfig, RpcConfig};
use crate::dashboard::DashboardState;
use crate::rpc::client::{HelixRpcConfig, UnifiedRpcClient};

/// High-level SDK facade for the HELIX protocol.
///
/// # Example
///
/// ```ignore
/// let client = HelixClient::from_profile(ConfigProfile::Local)?;
/// client.connect().await?;
/// ```
pub struct HelixClient {
    config: HelixConfig,
    rpc: UnifiedRpcClient,
    dashboard_state: Option<Arc<DashboardState>>,
}

impl HelixClient {
    /// Create a new `HelixClient` from a validated configuration.
    ///
    /// The RPC layer starts in mock-only mode. Call [`connect`](Self::connect)
    /// to upgrade to a real node connection.
    pub fn new(config: HelixConfig) -> Result<Self> {
        config.validate()?;
        let rpc = UnifiedRpcClient::mock_only();
        Ok(Self {
            config,
            rpc,
            dashboard_state: None,
        })
    }

    /// Load configuration from a TOML file and create the client.
    pub fn from_config_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let config = HelixConfig::load(&path)?;
        Self::new(config)
    }

    /// Create a client using a built-in configuration profile.
    pub fn from_profile(profile: ConfigProfile) -> Result<Self> {
        let config = HelixConfig::from_profile(profile);
        Self::new(config)
    }

    /// Attempt to connect the RPC layer to a real HELIX node.
    ///
    /// On failure the client falls back to mock mode automatically.
    pub async fn connect(&mut self) -> Result<()> {
        let rpc_cfg = rpc_config_from_helix(&self.config.rpc);
        self.rpc = UnifiedRpcClient::new(rpc_cfg).await;
        Ok(())
    }

    /// Attach a shared `DashboardState` to this client (builder-style).
    pub fn with_dashboard(mut self, state: Arc<DashboardState>) -> Self {
        self.dashboard_state = Some(state);
        self
    }

    // -- Accessors -----------------------------------------------------------

    /// Reference to the active configuration.
    pub fn config(&self) -> &HelixConfig {
        &self.config
    }

    /// Reference to the RPC client.
    pub fn rpc(&self) -> &UnifiedRpcClient {
        &self.rpc
    }

    /// The attached dashboard state, if any.
    pub fn dashboard_state(&self) -> Option<&Arc<DashboardState>> {
        self.dashboard_state.as_ref()
    }

    /// Whether the RPC layer is connected to a real node.
    pub fn is_connected(&self) -> bool {
        self.rpc.is_connected()
    }
}

/// Convert the crate-level `RpcConfig` into the rpc module's `HelixRpcConfig`.
fn rpc_config_from_helix(rpc: &RpcConfig) -> HelixRpcConfig {
    HelixRpcConfig {
        endpoint: rpc.url.clone(),
        timeout_secs: rpc.timeout,
        max_retries: rpc.max_retries,
        retry_delay_ms: rpc.retry_delay * 1000,
        max_backoff_ms: 10_000,
        compress: false,
        auth_token: None,
    }
}
