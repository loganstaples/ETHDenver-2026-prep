//! Init command implementation
//!
//! Handles node initialization including configuration generation, wallet creation,
//! directory setup, and network profile configuration.

use std::path::{Path, PathBuf};
use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use colored::*;
use serde::{Deserialize, Serialize};

use crate::config::{HelixConfig, NodeConfig, NetworkConfig, TrainingConfig, ContractConfig, RpcConfig, ConfigProfile, NodeRole};
use crate::wallet::{Wallet, WalletManager, WalletMetadata};
use crate::progress::ProgressDisplay;

/// Network profiles with pre-configured settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkProfile {
    /// Profile name
    pub name: String,
    /// RPC endpoint
    pub rpc_url: String,
    /// Chain ID
    pub chain_id: u64,
    /// Explorer URL
    pub explorer_url: Option<String>,
    /// Default gas price in gwei
    pub default_gas_price: u64,
    /// Contract addresses
    pub contracts: Option<ContractAddresses>,
    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,
}

impl NetworkProfile {
    /// Get the local/development profile
    pub fn local() -> Self {
        Self {
            name: "local".to_string(),
            rpc_url: "http://localhost:8545".to_string(),
            chain_id: 31337,
            explorer_url: None,
            default_gas_price: 1,
            contracts: None,
            bootstrap_nodes: vec![],
        }
    }

    /// Get the Anvil profile
    pub fn anvil() -> Self {
        Self {
            name: "anvil".to_string(),
            rpc_url: "http://127.0.0.1:8545".to_string(),
            chain_id: 31337,
            explorer_url: None,
            default_gas_price: 1,
            contracts: None,
            bootstrap_nodes: vec![],
        }
    }

    /// Get the Sepolia testnet profile
    pub fn sepolia() -> Self {
        Self {
            name: "sepolia".to_string(),
            rpc_url: "https://rpc.sepolia.org".to_string(),
            chain_id: 11155111,
            explorer_url: Some("https://sepolia.etherscan.io".to_string()),
            default_gas_price: 20,
            contracts: None,
            bootstrap_nodes: vec![
                "/ip4/34.220.195.23/tcp/9000/p2p/QmYyQSoG".to_string(),
            ],
        }
    }

    /// Get the Ethereum mainnet profile
    pub fn mainnet() -> Self {
        Self {
            name: "mainnet".to_string(),
            rpc_url: "https://eth.llamarpc.com".to_string(),
            chain_id: 1,
            explorer_url: Some("https://etherscan.io".to_string()),
            default_gas_price: 30,
            contracts: None,
            bootstrap_nodes: vec![
                "/ip4/52.88.123.45/tcp/9000/p2p/QmXoYPm".to_string(),
            ],
        }
    }

    /// Get profile by name
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "local" | "localhost" | "hardhat" => Some(Self::local()),
            "anvil" => Some(Self::anvil()),
            "sepolia" | "testnet" => Some(Self::sepolia()),
            "mainnet" => Some(Self::mainnet()),
            _ => None,
        }
    }
}

/// Contract addresses for a network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractAddresses {
    /// HelixCoordinator contract address
    pub coordinator: String,
    /// HelixVerifier contract address
    pub verifier: String,
    /// HelixToken contract address
    pub token: Option<String>,
    /// HelixStaking contract address
    pub staking: Option<String>,
}

/// Node initialization options
#[derive(Debug, Clone)]
pub struct InitOptions {
    /// Node name
    pub name: Option<String>,
    /// Network to initialize for
    pub network: String,
    /// Generate a new wallet
    pub generate_wallet: bool,
    /// Initialize as aggregator node
    pub aggregator: bool,
    /// Custom RPC URL
    pub rpc_url: Option<String>,
    /// Custom data directory
    pub data_dir: Option<PathBuf>,
    /// Custom config path
    pub config_path: Option<PathBuf>,
    /// Import existing wallet private key
    pub import_key: Option<String>,
    /// Skip confirmation prompts
    pub force: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            name: None,
            network: "local".to_string(),
            generate_wallet: false,
            aggregator: false,
            rpc_url: None,
            data_dir: None,
            config_path: None,
            import_key: None,
            force: false,
        }
    }
}

/// Initialization result containing all generated artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitResult {
    /// Node ID
    pub node_id: String,
    /// Node name
    pub node_name: String,
    /// Network
    pub network: String,
    /// Config file path
    pub config_path: PathBuf,
    /// Data directory
    pub data_dir: PathBuf,
    /// Wallet address (if generated)
    pub wallet_address: Option<String>,
    /// Wallet file path (if generated)
    pub wallet_path: Option<PathBuf>,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// RPC URL
    pub rpc_url: String,
    /// Chain ID
    pub chain_id: u64,
}

/// Init command handler
pub struct InitCommand {
    /// Progress display
    progress: ProgressDisplay,
}

impl InitCommand {
    /// Create a new init command
    pub fn new() -> Self {
        Self {
            progress: ProgressDisplay::new(),
        }
    }

    /// Execute the init command
    pub async fn execute(&mut self, options: InitOptions) -> Result<InitResult> {
        // Get HELIX home directory
        let helix_home = Self::get_helix_home()?;

        // Generate node name if not provided
        let node_name = options.name.unwrap_or_else(|| {
            format!("helix-node-{}", &uuid::Uuid::new_v4().to_string()[..8])
        });

        let node_id = uuid::Uuid::new_v4().to_string();

        self.progress.start_spinner(&format!("Initializing HELIX node '{}'...", node_name));

        // Get network profile
        let profile = NetworkProfile::by_name(&options.network)
            .ok_or_else(|| anyhow!("Unknown network: {}", options.network))?;

        // Setup directories
        let data_dir = options.data_dir.unwrap_or_else(|| helix_home.join("data").join(&node_name));
        let config_path = options.config_path.unwrap_or_else(|| helix_home.join(format!("{}.toml", node_name)));
        let wallets_dir = helix_home.join("wallets");
        let logs_dir = helix_home.join("logs");
        let cache_dir = helix_home.join("cache");

        // Create directories
        std::fs::create_dir_all(&data_dir).context("Failed to create data directory")?;
        std::fs::create_dir_all(&wallets_dir).context("Failed to create wallets directory")?;
        std::fs::create_dir_all(&logs_dir).context("Failed to create logs directory")?;
        std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;

        self.progress.finish_spinner("Directories created");

        // Determine capabilities
        let capabilities = if options.aggregator {
            vec!["train".to_string(), "aggregate".to_string(), "prove".to_string()]
        } else {
            vec!["train".to_string(), "prove".to_string()]
        };

        // Generate or import wallet
        let (wallet_address, wallet_path) = if options.generate_wallet || options.import_key.is_some() {
            self.progress.start_spinner("Setting up wallet...");

            let wallet = if let Some(key) = options.import_key {
                let pk = crate::wallet::PrivateKey::from_hex(&key)?;
                Wallet::from_private_key(&node_name, pk, &options.network)
            } else {
                Wallet::generate(&node_name, &options.network)
            };

            let addr = wallet.address();
            let wallet_file = wallets_dir.join(format!("{}.json", node_name));

            // Generate a secure random password for the keystore
            let mut pw_bytes = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut pw_bytes);
            let keystore_password = hex::encode(pw_bytes);

            // Save the password alongside the keystore for local dev convenience.
            // In production, use OS keychain via `wallet::keychain` instead.
            let pw_path = wallets_dir.join(format!("{}.password", node_name));
            std::fs::write(&pw_path, &keystore_password)?;

            // Save wallet to keystore
            wallet.save_keystore(&wallet_file, &keystore_password)?;

            self.progress.finish_spinner(&format!("Wallet created: {}", addr));

            (Some(addr.to_hex()), Some(wallet_file))
        } else {
            (None, None)
        };

        // Build configuration
        let rpc_url = options.rpc_url.unwrap_or(profile.rpc_url.clone());

        // Determine node role
        let role = if options.aggregator {
            NodeRole::Aggregator
        } else {
            NodeRole::Worker
        };

        // Build config using defaults and overriding specific fields
        let config = HelixConfig {
            profile: ConfigProfile::Local,
            node: NodeConfig {
                name: node_name.clone(),
                network: options.network.clone(),
                data_dir: data_dir.clone(),
                capabilities: capabilities.clone(),
                role,
                ..NodeConfig::default()
            },
            network: NetworkConfig {
                listen_addr: "0.0.0.0:9000".to_string(),
                bootstrap_nodes: profile.bootstrap_nodes.clone(),
                enable_mdns: options.network == "local",
                ..NetworkConfig::default()
            },
            training: TrainingConfig {
                batch_size: 32,
                learning_rate: 0.001,
                max_rounds: 100,
                ..TrainingConfig::default()
            },
            rpc: RpcConfig {
                url: rpc_url.clone(),
                chain_id: profile.chain_id,
                ..RpcConfig::default()
            },
            ..HelixConfig::default()
        };

        // Write configuration file
        self.progress.start_spinner("Writing configuration...");
        let config_str = toml::to_string_pretty(&config)?;
        std::fs::write(&config_path, config_str)?;
        self.progress.finish_spinner("Configuration written");

        // Write extended node config
        let extended_config = ExtendedNodeConfig {
            node_id: node_id.clone(),
            node_name: node_name.clone(),
            network: options.network.clone(),
            rpc_url: rpc_url.clone(),
            chain_id: profile.chain_id,
            capabilities: capabilities.clone(),
            data_dir: data_dir.clone(),
            wallets_dir: wallets_dir.clone(),
            logs_dir,
            cache_dir,
            wallet_address: wallet_address.clone(),
            contracts: profile.contracts.clone(),
            bootstrap_nodes: profile.bootstrap_nodes,
            created_at: chrono::Utc::now(),
        };

        let extended_path = helix_home.join(format!("{}_extended.json", node_name));
        let extended_json = serde_json::to_string_pretty(&extended_config)?;
        std::fs::write(&extended_path, extended_json)?;

        // Generate example training configuration
        self.progress.start_spinner("Creating example training config...");
        let training_config_path = helix_home.join("model.toml");
        if !training_config_path.exists() {
            let training_example = crate::config::training::TrainingJobConfig::example_toml();
            std::fs::write(&training_config_path, training_example)?;
        }
        self.progress.finish_spinner("Training config template created");

        Ok(InitResult {
            node_id,
            node_name,
            network: options.network,
            config_path,
            data_dir,
            wallet_address,
            wallet_path,
            capabilities,
            rpc_url,
            chain_id: profile.chain_id,
        })
    }

    /// Display the init result
    pub fn display_result(&self, result: &InitResult) {
        println!();
        println!("{}", "═".repeat(60).cyan());
        println!("{}", " HELIX Node Initialized Successfully".cyan().bold());
        println!("{}", "═".repeat(60).cyan());
        println!();

        println!("{}", "Node Information:".yellow().bold());
        println!("  Node ID:        {}", result.node_id.dimmed());
        println!("  Node Name:      {}", result.node_name.green());
        println!("  Network:        {}", result.network.cyan());
        println!("  Chain ID:       {}", result.chain_id);
        println!();

        println!("{}", "Paths:".yellow().bold());
        println!("  Config:         {}", result.config_path.display());
        println!("  Data Dir:       {}", result.data_dir.display());
        if let Some(ref wallet_path) = result.wallet_path {
            println!("  Wallet:         {}", wallet_path.display());
        }
        println!();

        println!("{}", "Network:".yellow().bold());
        println!("  RPC URL:        {}", result.rpc_url);
        if let Some(ref addr) = result.wallet_address {
            println!("  Wallet Address: {}", addr.green());
        }
        println!();

        println!("{}", "Capabilities:".yellow().bold());
        for cap in &result.capabilities {
            println!("  {} {}", "●".green(), cap);
        }
        println!();

        println!("{}", "Next Steps:".green().bold());
        println!("  1. Review configuration: {}", result.config_path.display());
        if result.wallet_address.is_some() {
            println!("  2. Fund your wallet with tokens for staking");
            println!("  3. Join a training network: helix join --coordinator <addr>");
        } else {
            println!("  2. Generate a wallet: helix init --generate-wallet");
            println!("  3. Join a training network: helix join --coordinator <addr>");
        }
        println!("  Or run a local demo: helix demo");
        println!();
    }

    /// Get the HELIX home directory
    fn get_helix_home() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("Could not determine home directory"))?;
        let helix_home = home.join(".helix");
        std::fs::create_dir_all(&helix_home)?;
        Ok(helix_home)
    }

    /// Check if already initialized
    pub fn is_initialized(name: &str) -> Result<bool> {
        let helix_home = Self::get_helix_home()?;
        let config_path = helix_home.join(format!("{}.toml", name));
        Ok(config_path.exists())
    }

    /// List all initialized nodes
    pub fn list_nodes() -> Result<Vec<NodeInfo>> {
        let helix_home = Self::get_helix_home()?;
        let mut nodes = Vec::new();

        for entry in std::fs::read_dir(&helix_home)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "toml") {
                if let Ok(config) = HelixConfig::load(&path) {
                    let extended_path = helix_home.join(format!("{}_extended.json", config.node.name));
                    let wallet_address = if let Ok(content) = std::fs::read_to_string(&extended_path) {
                        if let Ok(extended) = serde_json::from_str::<ExtendedNodeConfig>(&content) {
                            extended.wallet_address
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    nodes.push(NodeInfo {
                        name: config.node.name.clone(),
                        network: config.node.network.clone(),
                        config_path: path,
                        data_dir: config.node.data_dir.clone(),
                        capabilities: config.node.capabilities.clone(),
                        wallet_address,
                    });
                }
            }
        }

        Ok(nodes)
    }

    /// Delete a node configuration
    pub fn delete_node(name: &str, delete_data: bool) -> Result<()> {
        let helix_home = Self::get_helix_home()?;

        // Delete config file
        let config_path = helix_home.join(format!("{}.toml", name));
        if config_path.exists() {
            std::fs::remove_file(&config_path)?;
        }

        // Delete extended config
        let extended_path = helix_home.join(format!("{}_extended.json", name));
        if extended_path.exists() {
            std::fs::remove_file(&extended_path)?;
        }

        // Delete wallet
        let wallet_path = helix_home.join("wallets").join(format!("{}.json", name));
        if wallet_path.exists() {
            std::fs::remove_file(&wallet_path)?;
        }

        // Delete data directory if requested
        if delete_data {
            let data_dir = helix_home.join("data").join(name);
            if data_dir.exists() {
                std::fs::remove_dir_all(&data_dir)?;
            }
        }

        Ok(())
    }
}

impl Default for InitCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Extended node configuration stored as JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedNodeConfig {
    /// Node ID (UUID)
    pub node_id: String,
    /// Node name
    pub node_name: String,
    /// Network
    pub network: String,
    /// RPC URL
    pub rpc_url: String,
    /// Chain ID
    pub chain_id: u64,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// Data directory
    pub data_dir: PathBuf,
    /// Wallets directory
    pub wallets_dir: PathBuf,
    /// Logs directory
    pub logs_dir: PathBuf,
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Wallet address
    pub wallet_address: Option<String>,
    /// Contract addresses
    pub contracts: Option<ContractAddresses>,
    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Node information for listing
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node name
    pub name: String,
    /// Network
    pub network: String,
    /// Config file path
    pub config_path: PathBuf,
    /// Data directory
    pub data_dir: PathBuf,
    /// Capabilities
    pub capabilities: Vec<String>,
    /// Wallet address
    pub wallet_address: Option<String>,
}

/// Interactive init wizard
pub struct InitWizard {
    /// Collected options
    options: InitOptions,
}

impl InitWizard {
    /// Create a new wizard
    pub fn new() -> Self {
        Self {
            options: InitOptions::default(),
        }
    }

    /// Run the wizard (returns options, requires user interaction)
    pub fn collect_options(&mut self) -> &InitOptions {
        // In a real implementation, this would interactively prompt the user
        // For now, return defaults
        &self.options
    }

    /// Set name
    pub fn with_name(mut self, name: &str) -> Self {
        self.options.name = Some(name.to_string());
        self
    }

    /// Set network
    pub fn with_network(mut self, network: &str) -> Self {
        self.options.network = network.to_string();
        self
    }

    /// Enable wallet generation
    pub fn with_wallet(mut self) -> Self {
        self.options.generate_wallet = true;
        self
    }

    /// Set as aggregator
    pub fn as_aggregator(mut self) -> Self {
        self.options.aggregator = true;
        self
    }

    /// Build final options
    pub fn build(self) -> InitOptions {
        self.options
    }
}

impl Default for InitWizard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_profile_local() {
        let profile = NetworkProfile::local();
        assert_eq!(profile.chain_id, 31337);
        assert_eq!(profile.name, "local");
    }

    #[test]
    fn test_network_profile_by_name() {
        assert!(NetworkProfile::by_name("local").is_some());
        assert!(NetworkProfile::by_name("mainnet").is_some());
        assert!(NetworkProfile::by_name("unknown").is_none());
    }

    #[test]
    fn test_init_options_default() {
        let options = InitOptions::default();
        assert_eq!(options.network, "local");
        assert!(!options.generate_wallet);
        assert!(!options.aggregator);
    }

    #[test]
    fn test_init_wizard() {
        let options = InitWizard::new()
            .with_name("test-node")
            .with_network("sepolia")
            .with_wallet()
            .as_aggregator()
            .build();

        assert_eq!(options.name, Some("test-node".to_string()));
        assert_eq!(options.network, "sepolia");
        assert!(options.generate_wallet);
        assert!(options.aggregator);
    }
}
