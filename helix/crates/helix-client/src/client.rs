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
    /// The RPC layer starts disconnected. Call [`connect`](Self::connect)
    /// to establish a real node connection. This will NOT silently fall back
    /// to mock mode — use [`new_mock`](Self::new_mock) explicitly for testing.
    pub fn new(config: HelixConfig) -> Result<Self> {
        config.validate()?;
        let rpc = UnifiedRpcClient::new_disconnected();
        Ok(Self {
            config,
            rpc,
            dashboard_state: None,
        })
    }

    /// Create a `HelixClient` that explicitly runs in mock mode.
    ///
    /// Use this for development and testing when you don't have a real node
    /// available. Unlike [`new`](Self::new) + [`connect`](Self::connect),
    /// this never attempts a real connection.
    pub fn new_mock(config: HelixConfig) -> Result<Self> {
        config.validate()?;
        let rpc = UnifiedRpcClient::new_mock();
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
    /// Returns an error if the connection fails. For testing/demo scenarios
    /// where no real node is available, use [`new_mock`](Self::new_mock).
    pub async fn connect(&mut self) -> Result<()> {
        let rpc_cfg = rpc_config_from_helix(&self.config.rpc);
        self.rpc = UnifiedRpcClient::connect(rpc_cfg).await?;
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

    /// Attach an on-chain client by supplying a private key.
    ///
    /// Uses the RPC URL, chain ID, and coordinator address from this client's
    /// configuration to create and attach a `ChainClient`.
    #[cfg(feature = "chain")]
    pub async fn connect_chain(&mut self, private_key: &str) -> Result<()> {
        let chain = self.config.chain_client(private_key).await?;
        self.rpc.set_chain_client(chain);
        Ok(())
    }

    /// Mutable reference to the RPC client (for attaching chain client etc.).
    pub fn rpc_mut(&mut self) -> &mut crate::rpc::client::UnifiedRpcClient {
        &mut self.rpc
    }

    /// Run a full E2E training session using the `TrainingOrchestrator`.
    ///
    /// This orchestrates the complete flow:
    /// 1. Start Anvil (if `start_anvil` is set)
    /// 2. Deploy contracts (if `deploy_contracts` is set)
    /// 3. Register model + stake on-chain (if chain feature enabled)
    /// 4. Spawn aggregator + worker nodes
    /// 5. Run training rounds with proof submission
    /// 6. Clean up all processes
    ///
    /// After training completes, the RPC client is upgraded to connect to
    /// the real node (if it was started).
    pub async fn train(
        &mut self,
        orch_config: crate::orchestration::OrchestratorConfig,
    ) -> Result<crate::orchestration::TrainingResult> {
        let mut orchestrator = crate::orchestration::TrainingOrchestrator::new(orch_config)?;
        let result = orchestrator.train().await;

        // Always clean up processes, even on error
        if let Err(e) = orchestrator.shutdown().await {
            tracing::warn!("Shutdown error: {}", e);
        }

        // If deployment happened, update our config with new contract addresses
        if let Some(ref deployment) = result.as_ref().ok().and_then(|r| r.deployment.as_ref()) {
            self.config.contracts.coordinator = deployment.coordinator.clone();
            self.config.contracts.verifier = deployment.verifier.clone();
            if let Some(ref token) = deployment.token {
                self.config.contracts.token = Some(token.clone());
            }
            if let Some(ref staking) = deployment.staking {
                self.config.contracts.staking = Some(staking.clone());
            }
        }

        result
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
