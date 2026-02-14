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
use crate::error::HelixError;
use crate::model::{ModelHandle, SdkModelConfig, TrainingParams};
use crate::rpc::client::{
    HealthStatus, HelixRpcConfig, ModelInfo, ModelWeightsResponse, NetworkStatus, RoundInfo,
    TrainingHistory, TrainingReport, TrainingResultData, TrainingStatus, UnifiedRpcClient,
};
use crate::session::TrainingSession;

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

    // ==================== High-Level SDK API ====================

    /// Create a mock client and connect in one call (convenience for tests/demos).
    pub async fn connect_mock() -> Result<Self, HelixError> {
        let config = HelixConfig::from_profile(ConfigProfile::Local);
        let rpc = UnifiedRpcClient::new_mock();
        Ok(Self {
            config,
            rpc,
            dashboard_state: None,
        })
    }

    /// Create a client and connect to a specific URL in one call.
    pub async fn connect_to(url: &str) -> Result<Self, HelixError> {
        let rpc_cfg = HelixRpcConfig::with_endpoint(url);
        let rpc = UnifiedRpcClient::connect(rpc_cfg)
            .await
            .map_err(|e| HelixError::Connection(e.to_string()))?;
        let config = HelixConfig::from_profile(ConfigProfile::Local);
        Ok(Self {
            config,
            rpc,
            dashboard_state: None,
        })
    }

    /// Register a model using the SDK config type.
    pub async fn register_model(
        &self,
        config: SdkModelConfig,
    ) -> Result<ModelHandle, HelixError> {
        config.validate()?;

        let ipfs_hash = format!("Qm{:0>44}", hex::encode(&config.name));
        let model_id = self
            .rpc
            .register_model(&config.name, &ipfs_hash, config.min_stake)
            .await?;

        Ok(ModelHandle {
            model_id,
            name: config.name,
            commitment: format!("0x{}", "00".repeat(32)),
            architecture: config.architecture,
        })
    }

    /// Get model information by ID.
    pub async fn get_model(&self, model_id: u64) -> Result<ModelInfo, HelixError> {
        Ok(self.rpc.get_model(model_id).await?)
    }

    /// List all registered models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, HelixError> {
        Ok(self.rpc.list_models().await?)
    }

    /// Start a training session and return a streaming `TrainingSession`.
    pub async fn start_training(
        &self,
        model_id: u64,
        params: TrainingParams,
    ) -> Result<TrainingSession, HelixError> {
        self.rpc
            .start_training_for(model_id, params.rounds, params.round_duration_secs)
            .await?;

        Ok(TrainingSession::new(
            self.rpc.clone(),
            model_id,
            params,
        ))
    }

    /// Stake tokens for a model. Returns the mock/real transaction hash.
    pub async fn stake(
        &self,
        model_id: u64,
        amount_eth: f64,
    ) -> Result<String, HelixError> {
        Ok(self.rpc.stake(model_id, amount_eth).await?)
    }

    /// Unstake tokens for a model.
    pub async fn unstake(&self, model_id: u64) -> Result<String, HelixError> {
        Ok(self.rpc.unstake(model_id).await?)
    }

    /// Claim accumulated rewards.
    pub async fn claim_rewards(&self, model_id: u64) -> Result<String, HelixError> {
        Ok(self.rpc.claim_rewards(model_id).await?)
    }

    /// Get network status information.
    pub async fn node_status(&self) -> Result<NetworkStatus, HelixError> {
        Ok(self.rpc.get_network_status().await?)
    }

    /// Health check.
    pub async fn health(&self) -> Result<HealthStatus, HelixError> {
        Ok(self.rpc.health_check().await?)
    }

    /// Get the current training status from the connected node.
    pub async fn training_status(&self) -> Result<TrainingStatus, HelixError> {
        Ok(self.rpc.get_training_status().await?)
    }

    /// Get the training result for a completed round.
    pub async fn get_training_result(&self, round_id: u64) -> Result<TrainingResultData, HelixError> {
        Ok(self.rpc.get_training_result(round_id).await?)
    }

    /// Get the current round information for a model.
    pub async fn get_current_round(&self, model_id: u64) -> Result<RoundInfo, HelixError> {
        Ok(self.rpc.get_current_round(model_id).await?)
    }

    /// Get a specific round's information.
    pub async fn get_round(&self, model_id: u64, round_id: u64) -> Result<RoundInfo, HelixError> {
        Ok(self.rpc.get_round(model_id, round_id).await?)
    }

    // ==================== Result Distribution ====================

    /// Get model weights for a specific round (or latest if `round_id` is None).
    pub async fn get_model_weights(
        &self,
        model_id: u64,
        round_id: Option<u64>,
    ) -> Result<ModelWeightsResponse, HelixError> {
        Ok(self.rpc.get_model_weights(model_id, round_id).await?)
    }

    /// Download model weights to a local file path.
    ///
    /// Fetches the weights from the node and writes them to disk. Returns the
    /// commitment hash for optional verification.
    pub async fn download_model(
        &self,
        model_id: u64,
        round_id: Option<u64>,
        path: impl Into<PathBuf>,
    ) -> Result<String, HelixError> {
        let weights = self.rpc.get_model_weights(model_id, round_id).await?;
        let dest = path.into();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| HelixError::Other(anyhow::anyhow!("Failed to create directory: {}", e)))?;
        }
        std::fs::write(&dest, &weights.weight_bytes)
            .map_err(|e| HelixError::Other(anyhow::anyhow!("Failed to write model file: {}", e)))?;
        Ok(weights.commitment)
    }

    /// Download model weights, write to disk, and verify integrity.
    ///
    /// Fetches weights from the node, verifies the SHA-256 commitment matches,
    /// then writes to disk. Returns the full `ModelWeightsResponse` for metadata access.
    pub async fn download_model_verified(
        &self,
        model_id: u64,
        round_id: Option<u64>,
        path: impl Into<PathBuf>,
    ) -> Result<ModelWeightsResponse, HelixError> {
        let weights = self.rpc.get_model_weights(model_id, round_id).await?;

        // Verify integrity
        if !Self::verify_model(&weights.weight_bytes, &weights.commitment)? {
            return Err(HelixError::Other(anyhow::anyhow!(
                "Model integrity check failed: commitment mismatch"
            )));
        }

        let dest = path.into();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| HelixError::Other(anyhow::anyhow!("Failed to create directory: {}", e)))?;
        }
        std::fs::write(&dest, &weights.weight_bytes)
            .map_err(|e| HelixError::Other(anyhow::anyhow!("Failed to write model file: {}", e)))?;

        Ok(weights)
    }

    /// Verify that local model weights match the on-chain commitment.
    ///
    /// Computes the SHA-256 hash of the weight bytes and compares it to the
    /// commitment returned by the node.
    pub fn verify_model(weight_bytes: &[u8], expected_commitment: &str) -> Result<bool, HelixError> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(weight_bytes);
        let hash = hasher.finalize();
        let computed = format!("0x{}", hex::encode(hash));
        Ok(computed == expected_commitment)
    }

    /// Get full training history for a model.
    pub async fn get_training_history(
        &self,
        model_id: u64,
    ) -> Result<TrainingHistory, HelixError> {
        Ok(self.rpc.get_training_history(model_id).await?)
    }

    /// Get a full training report / certificate for a model.
    pub async fn get_training_report(
        &self,
        model_id: u64,
    ) -> Result<TrainingReport, HelixError> {
        Ok(self.rpc.get_training_report(model_id).await?)
    }

    /// Clone the underlying RPC client (for use by TrainingSession etc.).
    pub(crate) fn clone_rpc(&self) -> UnifiedRpcClient {
        self.rpc.clone()
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
