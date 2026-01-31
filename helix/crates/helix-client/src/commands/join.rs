//! Join command implementation
//!
//! Handles joining a HELIX training network, including coordinator connection,
//! staking tokens, peer discovery, and model synchronization.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use colored::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::progress::ProgressDisplay;
use crate::wallet::{Address, Balance, Wallet, Transaction, SignedTransaction};

/// Join command options
#[derive(Debug, Clone)]
pub struct JoinOptions {
    /// Coordinator address (contract or URL)
    pub coordinator: String,
    /// Model ID to join training for
    pub model_id: u64,
    /// Stake amount in ETH
    pub stake: Option<f64>,
    /// Node capabilities to advertise
    pub capabilities: Vec<String>,
    /// RPC endpoint override
    pub rpc_url: Option<String>,
    /// Wallet name to use
    pub wallet: Option<String>,
    /// Skip confirmation prompts
    pub force: bool,
    /// Connection timeout in seconds
    pub timeout: u64,
    /// Number of retries for failed operations
    pub retries: u32,
}

impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            coordinator: String::new(),
            model_id: 0,
            stake: None,
            capabilities: vec!["train".to_string(), "prove".to_string()],
            rpc_url: None,
            wallet: None,
            force: false,
            timeout: 30,
            retries: 3,
        }
    }
}

/// Result of joining a network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResult {
    /// Successfully joined
    pub success: bool,
    /// Model ID joined
    pub model_id: u64,
    /// Coordinator address
    pub coordinator: String,
    /// Worker address
    pub worker_address: String,
    /// Stake transaction hash
    pub stake_tx: Option<String>,
    /// Amount staked
    pub staked_amount: Option<f64>,
    /// Connected peers
    pub peers: Vec<PeerInfo>,
    /// Model state hash
    pub model_state: Option<String>,
    /// Current training round
    pub current_round: u64,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// Join timestamp
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

/// Information about a connected peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer ID
    pub id: String,
    /// Peer address
    pub address: String,
    /// Peer role
    pub role: PeerRole,
    /// Connection latency in ms
    pub latency_ms: u32,
    /// Peer reputation score
    pub reputation: f64,
}

/// Peer role in the network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerRole {
    /// Worker node (trains)
    Worker,
    /// Aggregator node
    Aggregator,
    /// Coordinator node
    Coordinator,
    /// Observer node
    Observer,
}

impl std::fmt::Display for PeerRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PeerRole::Worker => write!(f, "Worker"),
            PeerRole::Aggregator => write!(f, "Aggregator"),
            PeerRole::Coordinator => write!(f, "Coordinator"),
            PeerRole::Observer => write!(f, "Observer"),
        }
    }
}

/// Model information from coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: u64,
    /// Model name
    pub name: String,
    /// Model IPFS hash
    pub ipfs_hash: String,
    /// Current state commitment
    pub state_commitment: String,
    /// Current training round
    pub current_round: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Required stake amount
    pub min_stake: f64,
    /// Number of active workers
    pub active_workers: u32,
    /// Model owner address
    pub owner: String,
    /// Is training active
    pub is_active: bool,
}

/// Network connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Disconnected from network
    Disconnected,
    /// Connecting to coordinator
    Connecting,
    /// Connected, discovering peers
    PeerDiscovery,
    /// Syncing model state
    Syncing,
    /// Fully connected and ready
    Ready,
    /// Error state
    Error,
}

impl std::fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionState::Disconnected => write!(f, "Disconnected"),
            ConnectionState::Connecting => write!(f, "Connecting"),
            ConnectionState::PeerDiscovery => write!(f, "Discovering Peers"),
            ConnectionState::Syncing => write!(f, "Syncing"),
            ConnectionState::Ready => write!(f, "Ready"),
            ConnectionState::Error => write!(f, "Error"),
        }
    }
}

/// Join command handler
pub struct JoinCommand {
    /// Progress display
    progress: ProgressDisplay,
    /// Connection state
    state: Arc<RwLock<ConnectionState>>,
    /// Connected peers
    peers: Arc<RwLock<Vec<PeerInfo>>>,
}

impl JoinCommand {
    /// Create a new join command
    pub fn new() -> Self {
        Self {
            progress: ProgressDisplay::new(),
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            peers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Execute the join command
    pub async fn execute(&mut self, options: JoinOptions) -> Result<JoinResult> {
        self.set_state(ConnectionState::Connecting).await;

        println!("{}", "═".repeat(60).cyan());
        println!("{}", " Joining HELIX Training Network".cyan().bold());
        println!("{}", "═".repeat(60).cyan());
        println!();

        println!("{}", "Connection Details:".yellow().bold());
        println!("  Coordinator:  {}", options.coordinator.cyan());
        println!("  Model ID:     {}", options.model_id);
        if let Some(stake) = options.stake {
            println!("  Stake:        {} ETH", stake);
        }
        println!("  Capabilities: {:?}", options.capabilities);
        println!();

        // Step 1: Connect to coordinator
        self.progress.start_spinner("Connecting to coordinator...");
        let coordinator_info = self.connect_coordinator(&options).await?;
        self.progress.finish_spinner("Connected to coordinator");

        // Step 2: Query model information
        self.progress.start_spinner("Querying model information...");
        let model_info = self.query_model_info(&options, &coordinator_info).await?;
        self.progress.finish_spinner(&format!("Model '{}' found (Round {})", model_info.name, model_info.current_round));

        // Verify minimum stake requirement
        if let Some(stake) = options.stake {
            if stake < model_info.min_stake {
                return Err(anyhow!(
                    "Stake amount {} ETH is less than minimum {} ETH",
                    stake,
                    model_info.min_stake
                ));
            }
        }

        // Step 3: Stake tokens if required
        let stake_tx = if let Some(stake) = options.stake {
            self.progress.start_spinner(&format!("Staking {} ETH...", stake));
            let tx = self.stake_tokens(stake, &model_info, &options).await?;
            self.progress.finish_spinner(&format!("Staked {} ETH (tx: {}...)", stake, &tx[..10]));
            Some(tx)
        } else {
            None
        };

        // Step 4: Peer discovery
        self.set_state(ConnectionState::PeerDiscovery).await;
        self.progress.start_spinner("Discovering peers...");
        let peers = self.discover_peers(&options).await?;
        self.progress.finish_spinner(&format!("Discovered {} peers", peers.len()));

        *self.peers.write().await = peers.clone();

        // Step 5: Sync model state
        self.set_state(ConnectionState::Syncing).await;
        self.progress.start_spinner("Syncing model state...");
        let model_state = self.sync_model_state(&model_info).await?;
        self.progress.finish_spinner("Model state synced");

        // Step 6: Register as worker
        self.progress.start_spinner("Registering as worker...");
        let worker_address = self.register_worker(&model_info, &options).await?;
        self.progress.finish_spinner(&format!("Registered as worker: {}", worker_address));

        self.set_state(ConnectionState::Ready).await;

        let result = JoinResult {
            success: true,
            model_id: options.model_id,
            coordinator: options.coordinator.clone(),
            worker_address,
            stake_tx,
            staked_amount: options.stake,
            peers,
            model_state: Some(model_state),
            current_round: model_info.current_round,
            capabilities: options.capabilities.clone(),
            joined_at: chrono::Utc::now(),
        };

        Ok(result)
    }

    /// Display join result
    pub fn display_result(&self, result: &JoinResult) {
        println!();
        println!("{}", "═".repeat(60).green());
        println!("{}", " Successfully Joined Training Network!".green().bold());
        println!("{}", "═".repeat(60).green());
        println!();

        println!("{}", "Status:".yellow().bold());
        println!("  {} Ready to participate in training", "●".green());
        println!();

        println!("{}", "Network Info:".yellow().bold());
        println!("  Model ID:       {}", result.model_id);
        println!("  Current Round:  {}", result.current_round);
        println!("  Your Address:   {}", result.worker_address.cyan());
        println!();

        if let Some(staked) = result.staked_amount {
            println!("{}", "Staking:".yellow().bold());
            println!("  Amount Staked:  {} ETH", staked);
            if let Some(ref tx) = result.stake_tx {
                println!("  Transaction:    {}", tx);
            }
            println!();
        }

        println!("{}", "Connected Peers:".yellow().bold());
        println!("  ┌─────────────────────┬────────────┬──────────┐");
        println!("  │ Peer ID             │ Role       │ Latency  │");
        println!("  ├─────────────────────┼────────────┼──────────┤");
        for peer in &result.peers {
            println!(
                "  │ {:19} │ {:10} │ {:>6}ms │",
                &peer.id[..19.min(peer.id.len())],
                format!("{}", peer.role),
                peer.latency_ms
            );
        }
        println!("  └─────────────────────┴────────────┴──────────┘");
        println!();

        println!("{}", "Capabilities:".yellow().bold());
        for cap in &result.capabilities {
            println!("  {} {}", "●".green(), cap);
        }
        println!();

        println!("{}", "Next Steps:".green().bold());
        println!("  1. Monitor training: helix status --watch");
        println!("  2. View logs:        helix logs --follow");
        println!("  3. Check proofs:     helix query proof --model-id {}", result.model_id);
        println!();
    }

    /// Connect to the coordinator
    async fn connect_coordinator(&self, options: &JoinOptions) -> Result<CoordinatorInfo> {
        // Simulate connection with timeout
        let timeout = Duration::from_secs(options.timeout);
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Parse coordinator address
        let is_contract = options.coordinator.starts_with("0x");

        Ok(CoordinatorInfo {
            address: options.coordinator.clone(),
            is_contract,
            rpc_url: options.rpc_url.clone().unwrap_or_else(|| "http://localhost:8545".to_string()),
            chain_id: 31337,
            version: "1.0.0".to_string(),
        })
    }

    /// Query model information from coordinator
    async fn query_model_info(&self, options: &JoinOptions, coordinator: &CoordinatorInfo) -> Result<ModelInfo> {
        tokio::time::sleep(Duration::from_millis(300)).await;

        // Simulated model info
        Ok(ModelInfo {
            id: options.model_id,
            name: format!("helix-model-{}", options.model_id),
            ipfs_hash: "QmXoYPm8rPnxV3YHNqpGd8tVwL5c7sW9eZyMbTxNxNwXYZ".to_string(),
            state_commitment: format!("0x{}", hex::encode([0xAB; 32])),
            current_round: 42,
            total_rounds: 100,
            min_stake: 0.1,
            active_workers: 3,
            owner: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string(),
            is_active: true,
        })
    }

    /// Stake tokens for the model
    async fn stake_tokens(&self, amount: f64, model: &ModelInfo, options: &JoinOptions) -> Result<String> {
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Simulated transaction hash
        let tx_hash = format!("0x{}", hex::encode([0xDE; 32]));
        Ok(tx_hash)
    }

    /// Discover peers in the network
    async fn discover_peers(&self, options: &JoinOptions) -> Result<Vec<PeerInfo>> {
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Simulated peer discovery
        Ok(vec![
            PeerInfo {
                id: "helix-node-a1b2c3d4".to_string(),
                address: "0x742d35Cc6634C0532925a3b844Bc454e4438f44e".to_string(),
                role: PeerRole::Worker,
                latency_ms: 15,
                reputation: 0.98,
            },
            PeerInfo {
                id: "helix-node-e5f6g7h8".to_string(),
                address: "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string(),
                role: PeerRole::Worker,
                latency_ms: 23,
                reputation: 0.95,
            },
            PeerInfo {
                id: "helix-agg-01".to_string(),
                address: "0xdD2FD4581271e230360230F9337D5c0430Bf44C0".to_string(),
                role: PeerRole::Aggregator,
                latency_ms: 8,
                reputation: 1.0,
            },
        ])
    }

    /// Sync the model state
    async fn sync_model_state(&self, model: &ModelInfo) -> Result<String> {
        tokio::time::sleep(Duration::from_millis(800)).await;
        Ok(model.state_commitment.clone())
    }

    /// Register as a worker for the model
    async fn register_worker(&self, model: &ModelInfo, options: &JoinOptions) -> Result<String> {
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Generate a worker address
        let addr = format!("0x{}", hex::encode([0x12; 20]));
        Ok(addr)
    }

    /// Set connection state
    async fn set_state(&self, state: ConnectionState) {
        *self.state.write().await = state;
    }

    /// Get current connection state
    pub async fn get_state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Get connected peers
    pub async fn get_peers(&self) -> Vec<PeerInfo> {
        self.peers.read().await.clone()
    }

    /// Disconnect from network
    pub async fn disconnect(&mut self) -> Result<()> {
        self.set_state(ConnectionState::Disconnected).await;
        self.peers.write().await.clear();
        Ok(())
    }
}

impl Default for JoinCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordinator information
#[derive(Debug, Clone)]
struct CoordinatorInfo {
    /// Coordinator address (contract or URL)
    address: String,
    /// Is a contract address
    is_contract: bool,
    /// RPC URL
    rpc_url: String,
    /// Chain ID
    chain_id: u64,
    /// Coordinator version
    version: String,
}

/// Stake manager for handling staking operations
pub struct StakeManager {
    /// RPC URL
    rpc_url: String,
    /// Chain ID
    chain_id: u64,
    /// Staking contract address
    staking_contract: Option<String>,
}

impl StakeManager {
    /// Create a new stake manager
    pub fn new(rpc_url: &str, chain_id: u64) -> Self {
        Self {
            rpc_url: rpc_url.to_string(),
            chain_id,
            staking_contract: None,
        }
    }

    /// Set staking contract address
    pub fn with_contract(mut self, address: &str) -> Self {
        self.staking_contract = Some(address.to_string());
        self
    }

    /// Get current stake for an address and model
    pub async fn get_stake(&self, worker: &str, model_id: u64) -> Result<StakeInfo> {
        // Simulated stake info
        Ok(StakeInfo {
            model_id,
            worker: worker.to_string(),
            amount: Balance::from_ether(1.5),
            locked_until: chrono::Utc::now() + chrono::Duration::days(7),
            slashed: false,
            slashed_amount: Balance::from_wei(0),
            reputation: 0.98,
        })
    }

    /// Stake tokens
    pub async fn stake(&self, wallet: &Wallet, model_id: u64, amount: f64) -> Result<SignedTransaction> {
        let tx = Transaction::new()
            .with_gas_limit(100000)
            .with_value((amount * 1e18) as u128)
            .with_chain_id(self.chain_id);

        wallet.sign_transaction(tx)
    }

    /// Unstake tokens
    pub async fn unstake(&self, wallet: &Wallet, model_id: u64) -> Result<SignedTransaction> {
        let tx = Transaction::new()
            .with_gas_limit(80000)
            .with_chain_id(self.chain_id);

        wallet.sign_transaction(tx)
    }

    /// Claim rewards
    pub async fn claim_rewards(&self, wallet: &Wallet, model_id: u64) -> Result<SignedTransaction> {
        let tx = Transaction::new()
            .with_gas_limit(60000)
            .with_chain_id(self.chain_id);

        wallet.sign_transaction(tx)
    }
}

/// Stake information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeInfo {
    /// Model ID
    pub model_id: u64,
    /// Worker address
    pub worker: String,
    /// Staked amount
    pub amount: Balance,
    /// Lock expiration
    pub locked_until: chrono::DateTime<chrono::Utc>,
    /// Has been slashed
    pub slashed: bool,
    /// Amount slashed
    pub slashed_amount: Balance,
    /// Current reputation
    pub reputation: f64,
}

/// Network discovery manager
pub struct DiscoveryManager {
    /// Bootstrap nodes
    bootstrap_nodes: Vec<String>,
    /// Enable mDNS
    enable_mdns: bool,
    /// Discovery timeout
    timeout: Duration,
}

impl DiscoveryManager {
    /// Create a new discovery manager
    pub fn new() -> Self {
        Self {
            bootstrap_nodes: Vec::new(),
            enable_mdns: true,
            timeout: Duration::from_secs(30),
        }
    }

    /// Add bootstrap nodes
    pub fn with_bootstrap_nodes(mut self, nodes: Vec<String>) -> Self {
        self.bootstrap_nodes = nodes;
        self
    }

    /// Enable/disable mDNS
    pub fn with_mdns(mut self, enabled: bool) -> Self {
        self.enable_mdns = enabled;
        self
    }

    /// Set discovery timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Discover peers
    pub async fn discover(&self) -> Result<Vec<PeerInfo>> {
        // Simulated discovery
        tokio::time::sleep(Duration::from_millis(500)).await;

        let mut peers = Vec::new();

        // Try bootstrap nodes
        for (i, node) in self.bootstrap_nodes.iter().enumerate() {
            peers.push(PeerInfo {
                id: format!("bootstrap-{}", i),
                address: node.clone(),
                role: PeerRole::Coordinator,
                latency_ms: 20 + i as u32 * 5,
                reputation: 1.0,
            });
        }

        // mDNS discovered peers
        if self.enable_mdns {
            peers.push(PeerInfo {
                id: "mdns-local-1".to_string(),
                address: "/ip4/192.168.1.100/tcp/9000".to_string(),
                role: PeerRole::Worker,
                latency_ms: 5,
                reputation: 0.9,
            });
        }

        Ok(peers)
    }
}

impl Default for DiscoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_options_default() {
        let options = JoinOptions::default();
        assert!(options.coordinator.is_empty());
        assert_eq!(options.model_id, 0);
        assert!(options.stake.is_none());
    }

    #[test]
    fn test_peer_role_display() {
        assert_eq!(format!("{}", PeerRole::Worker), "Worker");
        assert_eq!(format!("{}", PeerRole::Aggregator), "Aggregator");
    }

    #[test]
    fn test_connection_state_display() {
        assert_eq!(format!("{}", ConnectionState::Ready), "Ready");
        assert_eq!(format!("{}", ConnectionState::Connecting), "Connecting");
    }

    #[tokio::test]
    async fn test_discovery_manager() {
        let manager = DiscoveryManager::new()
            .with_bootstrap_nodes(vec!["node1".to_string()])
            .with_mdns(true);

        let peers = manager.discover().await.unwrap();
        assert!(!peers.is_empty());
    }
}
