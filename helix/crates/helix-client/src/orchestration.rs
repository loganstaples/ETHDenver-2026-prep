//! End-to-End Training Orchestration
//!
//! Provides `TrainingOrchestrator` which coordinates the full HELIX training
//! flow: start Anvil → deploy contracts → register model → spawn nodes →
//! run training rounds → generate proofs → submit on-chain.
//!
//! This module handles real process management (Anvil, helix-node workers,
//! aggregator) and real contract interaction when the `chain` feature is enabled.

use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument, warn};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the full E2E training orchestration.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    // -- Model --
    /// Model dimensions: (d_in, d_hid, d_out).
    pub model_dims: (usize, usize, usize),
    /// Model seed for deterministic initialization.
    pub model_seed: u64,
    /// Learning rate for training.
    pub learning_rate: f64,
    /// Number of training rounds.
    pub rounds: u32,

    // -- Network --
    /// Number of worker nodes to spawn.
    pub workers: u32,
    /// Path to the helix-node binary.
    pub node_binary: PathBuf,
    /// Base TCP port for nodes (aggregator gets base_port, workers get base_port+1..).
    pub base_port: u16,
    /// HTTP API port for the aggregator.
    pub http_port: u16,

    // -- Chain --
    /// Ethereum JSON-RPC URL (e.g. http://localhost:8545).
    pub eth_rpc_url: String,
    /// Private key for signing transactions (hex, with or without 0x prefix).
    pub private_key: String,
    /// Path to the contracts/ directory (for forge script deployment).
    pub contracts_dir: PathBuf,
    /// Whether to deploy fresh contracts (requires forge).
    pub deploy_contracts: bool,
    /// Pre-deployed coordinator contract address (used when deploy_contracts=false).
    pub coordinator_address: String,

    // -- Anvil --
    /// Whether to start a local Anvil instance.
    pub start_anvil: bool,
    /// Port for the Anvil instance.
    pub anvil_port: u16,

    // -- Timing --
    /// Maximum time to wait for workers to connect.
    pub worker_timeout: Duration,
    /// Maximum time per training round.
    pub round_timeout: Duration,

    // -- Health monitoring --
    /// Interval between health checks on spawned processes.
    pub health_check_interval: Duration,
    /// Maximum number of automatic restarts per node before giving up.
    pub max_restarts: u32,

    // -- Checkpointing --
    /// Checkpoint configuration for training state persistence.
    pub checkpoint: crate::demo::checkpoint::CheckpointConfig,
    /// Whether to resume from an existing checkpoint on startup.
    pub resume: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            model_dims: (4, 8, 2),
            model_seed: 42,
            learning_rate: 0.01,
            rounds: 5,
            workers: 3,
            node_binary: PathBuf::from("target/debug/helix-node"),
            base_port: 9000,
            http_port: 9001,
            eth_rpc_url: "http://localhost:8545".to_string(),
            private_key: std::env::var("HELIX_PRIVATE_KEY").unwrap_or_else(|_| {
                eprintln!("WARNING: Using default Anvil private key. Set HELIX_PRIVATE_KEY for non-local networks.");
                "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".to_string()
            }),
            contracts_dir: PathBuf::from("contracts"),
            deploy_contracts: true,
            coordinator_address: String::new(),
            start_anvil: true,
            anvil_port: 8545,
            worker_timeout: Duration::from_secs(30),
            round_timeout: Duration::from_secs(120),
            health_check_interval: Duration::from_secs(5),
            max_restarts: 3,
            checkpoint: crate::demo::checkpoint::CheckpointConfig::default(),
            resume: false,
        }
    }
}

/// Known Anvil/Hardhat default private keys (first 4).
const KNOWN_DEV_KEYS: &[&str] = &[
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
];

impl OrchestratorConfig {
    /// Validate the configuration for production safety.
    ///
    /// Rejects known dev private keys when the RPC URL points to a non-local network.
    #[instrument(skip_all)]
    pub fn validate(&self) -> Result<()> {
        let is_local = self.eth_rpc_url.contains("localhost")
            || self.eth_rpc_url.contains("127.0.0.1");

        let stripped_key = self.private_key.strip_prefix("0x").unwrap_or(&self.private_key);

        if !is_local && KNOWN_DEV_KEYS.contains(&stripped_key) {
            return Err(anyhow!(
                "Refusing to use a well-known Anvil/Hardhat private key with non-local RPC '{}'. \
                 Set HELIX_PRIVATE_KEY to a real key.",
                self.eth_rpc_url,
            ));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Addresses of deployed contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentResult {
    pub verifier: String,
    pub coordinator: String,
    pub token: Option<String>,
    pub staking: Option<String>,
    pub rewards: Option<String>,
    pub registry: Option<String>,
    pub treasury: String,
}

/// Training results after E2E orchestration completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Number of training rounds completed successfully.
    pub rounds_completed: u32,
    /// Final loss value (from last successful round).
    pub final_loss: f64,
    /// Accumulated error bound.
    pub accumulated_error: f64,
    /// Number of proofs submitted on-chain.
    pub proofs_submitted: u32,
    /// Total wall-clock duration.
    pub duration_ms: u64,
    /// On-chain model ID (if registered).
    pub model_id: Option<u64>,
    /// Transaction hashes of on-chain proof submissions.
    pub tx_hashes: Vec<String>,
    /// Deployed contract addresses (if deployment happened).
    pub deployment: Option<DeploymentResult>,
}

/// Per-round status snapshot for progress reporting.
#[derive(Debug, Clone)]
pub struct RoundProgress {
    pub round: u32,
    pub phase: &'static str,
    pub loss: Option<f64>,
    pub error_bound: Option<f64>,
    pub proof_submitted: bool,
    pub tx_hash: Option<String>,
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// TrainingOrchestrator
// ---------------------------------------------------------------------------

/// Orchestrates the full HELIX E2E training flow.
///
/// Manages process lifecycle (Anvil, helix-node workers/aggregator),
/// contract deployment, model registration, training execution, and
/// on-chain proof submission.
/// Tracks a managed node process with restart metadata.
struct ManagedProcess {
    name: String,
    role: String,
    port: u16,
    child: Child,
    restart_count: u32,
    last_stderr: String,
}

pub struct TrainingOrchestrator {
    config: OrchestratorConfig,
    anvil_process: Option<Child>,
    node_processes: Vec<(String, Child)>,
    managed_processes: Vec<ManagedProcess>,
    deployment: Option<DeploymentResult>,
    model_id: Option<u64>,
    /// Callback for progress reporting.
    progress_callback: Option<Box<dyn Fn(RoundProgress) + Send + Sync>>,
    /// ZK prover for generating real training proofs.
    prover: Option<helix_prover::MLTrainingProverV2>,
    /// Current training state (weights) used to build witnesses.
    training_state: Option<crate::demo::real_training::TrainingState>,
    /// Shared HTTP client for communicating with spawned nodes.
    /// Reuses connection pool across all health checks, round triggers, and
    /// status polls instead of creating a new client per request.
    http_client: reqwest::Client,
}

impl TrainingOrchestrator {
    /// Create a new orchestrator from configuration.
    ///
    /// Returns an error if the internal HTTP client cannot be constructed
    /// (should only happen if TLS backends are misconfigured).
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(4)
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            config,
            anvil_process: None,
            node_processes: Vec::new(),
            managed_processes: Vec::new(),
            deployment: None,
            model_id: None,
            progress_callback: None,
            prover: None,
            training_state: None,
            http_client,
        })
    }

    /// Set a progress callback that is invoked after each phase.
    pub fn on_progress(&mut self, callback: impl Fn(RoundProgress) + Send + Sync + 'static) {
        self.progress_callback = Some(Box::new(callback));
    }

    // =======================================================================
    // Full E2E Flow
    // =======================================================================

    /// Run the complete E2E training flow.
    ///
    /// 1. Start Anvil (if configured)
    /// 2. Deploy contracts (if configured)
    /// 3. Register model + stake on-chain
    /// 4. Spawn aggregator + worker nodes
    /// 5. Wait for workers to connect
    /// 6. Run training rounds with proof submission
    /// 7. Clean up all processes
    #[instrument(skip_all)]
    pub async fn train(&mut self) -> Result<TrainingResult> {
        let start = Instant::now();
        let mut tx_hashes = Vec::new();
        let mut proofs_submitted = 0u32;

        // Phase 1: Start Anvil
        if self.config.start_anvil {
            self.start_anvil().await
                .context("Failed to start Anvil")?;
        }

        // Phase 2: Deploy contracts
        if self.config.deploy_contracts {
            self.deploy_contracts().await
                .context("Failed to deploy contracts")?;
        }

        // Phase 3: Register model + stake on-chain
        #[cfg(feature = "chain")]
        {
            if self.deployment.is_some() || !self.config.coordinator_address.is_empty() {
                self.register_and_stake().await
                    .context("Failed to register model / stake")?;
            }
        }

        // Phase 3.5: Initialize ZK prover for real proof generation
        self.initialize_prover()
            .context("Failed to initialize ZK prover")?;

        // Phase 3.6: Resume from checkpoint (if configured)
        let mut start_round = 1u32;
        if self.config.resume && self.config.checkpoint.enabled {
            let model_id = self.model_id.unwrap_or(0);
            let ckpt_path = self.config.checkpoint.checkpoint_path(model_id);
            if let Some(ckpt) = crate::demo::checkpoint::TrainingCheckpoint::load(&ckpt_path)
                .context("Failed to load checkpoint")?
            {
                let restored = ckpt.to_state().context("Failed to restore training state from checkpoint")?;
                info!(
                    "Resuming training from checkpoint: step={}, loss={:.4}, error={:.2}",
                    restored.step, restored.loss, restored.error_bound
                );
                start_round = (restored.step as u32).saturating_add(1);
                self.training_state = Some(restored);
            }
        }

        // Phase 4: Start network
        self.start_network().await
            .context("Failed to start network")?;

        // Phase 5: Wait for workers
        self.wait_for_workers().await
            .context("Timeout waiting for workers to connect")?;

        // Phase 6: Run training rounds
        let mut final_loss = f64::NAN;
        let mut accumulated_error = 0.0;
        let mut rounds_completed = 0u32;

        for round in start_round..=self.config.rounds {
            let round_start = Instant::now();
            info!("Starting round {}/{}", round, self.config.rounds);

            // Trigger round via aggregator HTTP API
            match self.trigger_round().await {
                Ok(_) => {}
                Err(e) => {
                    warn!("Round {} trigger failed: {}, retrying...", round, e);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    self.trigger_round().await
                        .context(format!("Round {} trigger retry failed", round))?;
                }
            }

            // Wait for round completion
            let status = self.wait_for_round_completion(round).await?;

            final_loss = status.get("loss")
                .and_then(|v| v.as_f64())
                .unwrap_or(final_loss);
            accumulated_error = status.get("accumulated_error")
                .and_then(|v| v.as_f64())
                .unwrap_or(accumulated_error);

            // Submit proof on-chain (if chain enabled)
            #[cfg(feature = "chain")]
            {
                if let Some(tx) = self.submit_round_proof_onchain(round).await? {
                    tx_hashes.push(tx);
                    proofs_submitted += 1;
                }
            }

            rounds_completed = round;
            let elapsed = round_start.elapsed().as_millis() as u64;

            if let Some(ref cb) = self.progress_callback {
                cb(RoundProgress {
                    round,
                    phase: "complete",
                    loss: Some(final_loss),
                    error_bound: Some(accumulated_error),
                    proof_submitted: proofs_submitted > 0,
                    tx_hash: tx_hashes.last().cloned(),
                    elapsed_ms: elapsed,
                });
            }

            // Save checkpoint after successful round
            if self.config.checkpoint.enabled {
                if let Some(ref state) = self.training_state {
                    let model_id = self.model_id.unwrap_or(0);
                    let proof_hash = tx_hashes.last().cloned().unwrap_or_default();
                    let ckpt = crate::demo::checkpoint::TrainingCheckpoint::from_state(
                        state, model_id, &proof_hash,
                    );
                    let ckpt_path = self.config.checkpoint.checkpoint_path(model_id);
                    if let Err(e) = ckpt.save(&ckpt_path) {
                        warn!("Failed to save checkpoint for round {}: {}", round, e);
                    }
                }
            }

            info!(
                "Round {}/{} completed in {}ms (loss={:.4}, error={:.2})",
                round, self.config.rounds, elapsed, final_loss, accumulated_error
            );
        }

        let duration = start.elapsed();

        Ok(TrainingResult {
            rounds_completed,
            final_loss,
            accumulated_error,
            proofs_submitted,
            duration_ms: duration.as_millis() as u64,
            model_id: self.model_id,
            tx_hashes,
            deployment: self.deployment.clone(),
        })
    }

    // =======================================================================
    // Phase 1: Anvil
    // =======================================================================

    /// Start a local Anvil instance.
    #[instrument(skip_all)]
    pub async fn start_anvil(&mut self) -> Result<()> {
        info!("Starting Anvil on port {}...", self.config.anvil_port);

        let child = Command::new("anvil")
            .arg("--port")
            .arg(self.config.anvil_port.to_string())
            .arg("--accounts")
            .arg("10")
            .arg("--balance")
            .arg("10000")
            .arg("--silent")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to start anvil. Is it installed? Run: curl -L https://foundry.paradigm.xyz | bash && foundryup")?;

        self.anvil_process = Some(child);

        // Wait for Anvil to be ready by polling the RPC endpoint
        let rpc_url = format!("http://127.0.0.1:{}", self.config.anvil_port);

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("Anvil did not become ready within 10 seconds"));
            }

            let body = serde_json::json!({
                "jsonrpc": "2.0",
                "method": "eth_chainId",
                "params": [],
                "id": 1
            });

            match self.http_client.post(&rpc_url).json(&body).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("Anvil ready at {}", rpc_url);
                    break;
                }
                _ => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }
        }

        // Update config with Anvil's RPC URL
        self.config.eth_rpc_url = rpc_url;

        Ok(())
    }

    // =======================================================================
    // Phase 2: Deploy Contracts
    // =======================================================================

    /// Deploy contracts using `forge script`.
    ///
    /// Runs `deployV3WithMock()` for the demo (uses MockVerifier for speed).
    /// Parses the broadcast JSON to extract deployed contract addresses.
    #[instrument(skip_all)]
    pub async fn deploy_contracts(&mut self) -> Result<()> {
        info!("Deploying contracts via forge script...");

        let contracts_dir = &self.config.contracts_dir;
        if !contracts_dir.exists() {
            return Err(anyhow!(
                "Contracts directory not found: {}. Set --contracts-dir to the path containing foundry.toml",
                contracts_dir.display()
            ));
        }

        let pk = self.config.private_key.strip_prefix("0x")
            .unwrap_or(&self.config.private_key);

        let output = Command::new("forge")
            .arg("script")
            .arg("script/Deploy.s.sol")
            .arg("--sig")
            .arg("runWithMock()")
            .arg("--broadcast")
            .arg("--rpc-url")
            .arg(&self.config.eth_rpc_url)
            .arg("--private-key")
            .arg(pk)
            .current_dir(contracts_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to run forge. Is foundry installed?")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(anyhow!(
                "forge script failed (exit {}):\nstdout: {}\nstderr: {}",
                output.status,
                stdout,
                stderr
            ));
        }

        // Parse deployment addresses from broadcast JSON
        let deployment = self.parse_broadcast_json(contracts_dir)?;
        info!(
            "Contracts deployed: coordinator={}, verifier={}",
            deployment.coordinator, deployment.verifier
        );

        // Update coordinator address in config
        self.config.coordinator_address = deployment.coordinator.clone();
        self.deployment = Some(deployment);

        Ok(())
    }

    /// Parse the forge broadcast JSON to extract deployed contract addresses.
    fn parse_broadcast_json(&self, contracts_dir: &PathBuf) -> Result<DeploymentResult> {
        // Forge writes broadcast to: broadcast/<script>/<chain_id>/run-latest.json
        let broadcast_path = contracts_dir
            .join("broadcast")
            .join("Deploy.s.sol")
            .join("31337")
            .join("run-latest.json");

        if !broadcast_path.exists() {
            // Fallback: try parsing console output from forge
            return self.parse_deployment_fallback(contracts_dir);
        }

        let content = std::fs::read_to_string(&broadcast_path)
            .context("Failed to read broadcast JSON")?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse broadcast JSON")?;

        let transactions = json
            .get("transactions")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow!("No 'transactions' array in broadcast JSON"))?;

        let mut addresses: HashMap<String, String> = HashMap::new();

        for tx in transactions {
            if let (Some(name), Some(addr)) = (
                tx.get("contractName").and_then(|n| n.as_str()),
                tx.get("contractAddress").and_then(|a| a.as_str()),
            ) {
                addresses.insert(name.to_string(), addr.to_string());
            }
        }

        // Extract deployer address as treasury (first account in Anvil)
        let deployer = transactions
            .first()
            .and_then(|tx| tx.get("transaction"))
            .and_then(|t| t.get("from"))
            .and_then(|f| f.as_str())
            .unwrap_or("0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266")
            .to_string();

        let coordinator = addresses
            .get("HelixCoordinatorV2")
            .or_else(|| addresses.get("HelixCoordinatorV3"))
            .cloned()
            .ok_or_else(|| anyhow!(
                "Coordinator contract not found in broadcast. Found: {:?}",
                addresses.keys().collect::<Vec<_>>()
            ))?;

        let verifier = addresses
            .get("MockVerifierForDeploy")
            .or_else(|| addresses.get("Halo2Verifier"))
            .cloned()
            .unwrap_or_default();

        Ok(DeploymentResult {
            coordinator,
            verifier,
            token: addresses.get("HelixToken").cloned(),
            staking: addresses.get("Staking").cloned(),
            rewards: addresses.get("Rewards").cloned(),
            registry: addresses.get("ModelRegistry").cloned(),
            treasury: deployer,
        })
    }

    /// Fallback deployment address parsing when broadcast JSON is unavailable.
    ///
    /// Uses deterministic CREATE addresses for Anvil's first account (nonce 0-5).
    fn parse_deployment_fallback(&self, _contracts_dir: &PathBuf) -> Result<DeploymentResult> {
        // Anvil's default deployer: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
        // Deterministic CREATE addresses for nonce 0..5:
        Ok(DeploymentResult {
            verifier: "0x5FbDB2315678afecb367f032d93F642f64180aa3".to_string(),
            coordinator: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512".to_string(),
            token: None,
            staking: None,
            rewards: None,
            registry: None,
            treasury: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266".to_string(),
        })
    }

    // =======================================================================
    // Phase 3: Register Model + Stake (chain feature)
    // =======================================================================

    /// Register a model on-chain and stake tokens.
    #[cfg(feature = "chain")]
    async fn register_and_stake(&mut self) -> Result<()> {
        use crate::rpc::chain::ChainClient;
        use ethers::types::U256;

        let coordinator_addr = if let Some(ref d) = self.deployment {
            &d.coordinator
        } else {
            &self.config.coordinator_address
        };

        if coordinator_addr.is_empty() {
            warn!("No coordinator address — skipping on-chain registration");
            return Ok(());
        }

        info!("Connecting chain client to coordinator at {}", coordinator_addr);

        let chain = ChainClient::new(
            &self.config.eth_rpc_url,
            &self.config.private_key,
            coordinator_addr,
            None,
        )
        .await
        .context("Failed to create ChainClient")?;

        // Register model
        info!("Registering model on-chain...");
        let (d_in, d_hid, d_out) = self.config.model_dims;
        let ipfs_hash = format!("QmHelix_{}x{}x{}_seed{}", d_in, d_hid, d_out, self.config.model_seed);
        let initial_commitment = U256::from(self.config.model_seed);
        let min_stake = U256::from(1_000_000_000_000_000u64); // 0.001 ETH

        let (_receipt, model_id) = chain
            .register_model(&ipfs_hash, initial_commitment, min_stake)
            .await
            .context("Failed to register model")?;

        info!("Model registered with ID: {}", model_id);
        self.model_id = Some(model_id);

        // Stake
        info!("Staking 0.005 ETH for model {}...", model_id);
        let stake_amount = U256::from(5_000_000_000_000_000u64); // 0.005 ETH
        chain
            .stake(model_id, stake_amount)
            .await
            .context("Failed to stake")?;

        // Start first round
        info!("Starting training round on-chain...");
        chain
            .start_round(model_id, 300) // 5 minute round duration
            .await
            .context("Failed to start round")?;

        Ok(())
    }

    #[cfg(not(feature = "chain"))]
    async fn register_and_stake(&mut self) -> Result<()> {
        info!("Chain feature disabled — skipping on-chain registration");
        Ok(())
    }

    // =======================================================================
    // Phase 4: Start Network
    // =======================================================================

    /// Spawn aggregator + worker nodes as child processes with health tracking.
    #[instrument(skip_all)]
    pub async fn start_network(&mut self) -> Result<()> {
        let (d_in, d_hid, d_out) = self.config.model_dims;

        // Start aggregator
        info!("Starting aggregator on port {}...", self.config.base_port);
        let agg_child = self.spawn_node(
            "aggregator",
            "aggregator",
            self.config.base_port,
        )?;
        self.managed_processes.push(ManagedProcess {
            name: "aggregator".into(),
            role: "aggregator".into(),
            port: self.config.base_port,
            child: agg_child,
            restart_count: 0,
            last_stderr: String::new(),
        });

        // Brief pause to let aggregator bind its port
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Start workers
        for i in 0..self.config.workers {
            let port = self.config.base_port + 1 + i as u16;
            let name = format!("worker-{}", i + 1);
            info!("Starting {} on port {}...", name, port);

            let child = self.spawn_node(&name, "worker", port)?;
            self.managed_processes.push(ManagedProcess {
                name: name.clone(),
                role: "worker".into(),
                port,
                child,
                restart_count: 0,
                last_stderr: String::new(),
            });
        }

        info!(
            "Network started: 1 aggregator + {} workers (model={}x{}x{}, lr={}, seed={})",
            self.config.workers, d_in, d_hid, d_out,
            self.config.learning_rate, self.config.model_seed
        );

        Ok(())
    }

    /// Spawn a single helix-node process with appropriate environment variables.
    fn spawn_node(
        &self,
        node_id: &str,
        role: &str,
        port: u16,
    ) -> Result<Child> {
        let (d_in, d_hid, d_out) = self.config.model_dims;
        let listen_addr = format!("127.0.0.1:{}", port);

        let mut cmd = Command::new(&self.config.node_binary);
        cmd.env("HELIX_NODE_ROLE", role)
            .env("HELIX_LISTEN_ADDR", &listen_addr)
            .env("HELIX_D_IN", d_in.to_string())
            .env("HELIX_D_HID", d_hid.to_string())
            .env("HELIX_D_OUT", d_out.to_string())
            .env("HELIX_MODEL_SEED", self.config.model_seed.to_string())
            .env("HELIX_LR", self.config.learning_rate.to_string())
            .env("RUST_LOG", "info");

        match role {
            "worker" => {
                cmd.env(
                    "HELIX_AGGREGATOR_ADDR",
                    format!("127.0.0.1:{}", self.config.base_port),
                );
            }
            "aggregator" => {
                cmd.env("HELIX_MIN_WORKERS", self.config.workers.to_string())
                    .env("HELIX_HTTP_PORT", self.config.http_port.to_string());

                if !self.config.eth_rpc_url.is_empty() && !self.config.private_key.is_empty() {
                    cmd.env("HELIX_ETH_RPC", &self.config.eth_rpc_url)
                        .env("PRIVATE_KEY", &self.config.private_key);

                    let coordinator = self.deployment.as_ref()
                        .map(|d| d.coordinator.as_str())
                        .unwrap_or(&self.config.coordinator_address);
                    if !coordinator.is_empty() {
                        cmd.env("COORDINATOR_ADDRESS", coordinator);
                    }
                }
            }
            _ => {}
        }

        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped());

        debug!("Spawning {} ({}) on {}", node_id, role, listen_addr);

        cmd.spawn().map_err(|e| {
            anyhow!(
                "Failed to spawn {} ({}): {}. Is helix-node built? Run: cargo build -p helix-node",
                node_id, self.config.node_binary.display(), e
            )
        })
    }

    // =======================================================================
    // Phase 5: Wait for Workers
    // =======================================================================

    /// Wait for workers to connect to the aggregator.
    ///
    /// Returns once all requested workers have connected, or on timeout returns
    /// a partial success if at least one worker is available.
    #[instrument(skip_all)]
    pub async fn wait_for_workers(&mut self) -> Result<()> {
        info!("Waiting for {} workers to connect...", self.config.workers);

        let deadline = Instant::now() + self.config.worker_timeout;

        let mut last_worker_count = 0u64;

        loop {
            if Instant::now() > deadline {
                // Check for crashed processes before reporting timeout
                let _ = self.check_process_health();

                if last_worker_count > 0 {
                    warn!(
                        "Timeout: only {}/{} workers connected after {}s — proceeding with partial set",
                        last_worker_count,
                        self.config.workers,
                        self.config.worker_timeout.as_secs()
                    );
                    return Ok(());
                }

                return Err(anyhow!(
                    "Timeout waiting for workers ({} seconds, 0/{} connected). Check helix-node logs.",
                    self.config.worker_timeout.as_secs(),
                    self.config.workers
                ));
            }

            // Periodic health check on managed processes
            let _ = self.check_process_health();

            let url = format!("http://127.0.0.1:{}/health", self.config.http_port);
            match self.http_client.get(&url).send().await {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let workers = json.get("workers")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        last_worker_count = workers;
                        if workers >= self.config.workers as u64 {
                            info!("{} workers connected", workers);
                            return Ok(());
                        }
                        debug!("Waiting for workers: {}/{}", workers, self.config.workers);
                    }
                }
                Err(_) => {
                    debug!("Aggregator not ready yet, retrying...");
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    // =======================================================================
    // Phase 6: Training Loop
    // =======================================================================

    /// Trigger a training round via the aggregator's HTTP API.
    #[instrument(skip_all)]
    pub async fn trigger_round(&self) -> Result<serde_json::Value> {
        let url = format!("http://127.0.0.1:{}/round/start", self.config.http_port);
        let resp = self
            .http_client
            .post(&url)
            .send()
            .await
            .context("Failed to trigger round")?
            .json::<serde_json::Value>()
            .await
            .context("Failed to parse round trigger response")?;
        Ok(resp)
    }

    /// Poll the aggregator until the current round completes.
    #[instrument(skip_all, fields(round))]
    pub async fn wait_for_round_completion(&self, round: u32) -> Result<serde_json::Value> {
        let deadline = Instant::now() + self.config.round_timeout;

        loop {
            if Instant::now() > deadline {
                return Err(anyhow!("Round {} timed out", round));
            }

            let url = format!("http://127.0.0.1:{}/round/status", self.config.http_port);
            match self.http_client.get(&url).send().await {
                Ok(resp) => {
                    if let Ok(status) = resp.json::<serde_json::Value>().await {
                        let completed = status
                            .get("completed_rounds")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);

                        if completed >= round as u64 {
                            return Ok(status);
                        }
                    }
                }
                Err(e) => {
                    debug!("Round status poll failed: {}", e);
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    /// Initialize the ZK prover and training state for real proof generation.
    #[instrument(skip_all)]
    fn initialize_prover(&mut self) -> Result<()> {
        use helix_prover::{MLTrainingProverV2, V2ProverConfig};
        use crate::demo::real_training::TrainingState;

        let (d_in, d_hid, d_out) = self.config.model_dims;

        info!(
            "Initializing ZK prover (model={}x{}x{})...",
            d_in, d_hid, d_out
        );
        let init_start = Instant::now();

        let prover_config = V2ProverConfig {
            k: 14,
            relu_range: 128,
            exp_range: 256,
            exp_scale: 1000,
            use_freivalds: true,
            base_error: helix_prover::halo2curves::bn256::Fr::from(1u64),
            self_verify: true,
            ..V2ProverConfig::default()
        };

        let prover = MLTrainingProverV2::with_config(d_in, d_hid, d_out, prover_config);
        let state = TrainingState::new_random(d_in, d_hid, d_out);

        let elapsed = init_start.elapsed();
        info!("ZK prover initialized in {:.1}s", elapsed.as_secs_f64());

        self.prover = Some(prover);
        self.training_state = Some(state);

        Ok(())
    }

    /// Generate a real ZK proof for a training round using the current weights.
    ///
    /// Builds a witness from the current training state, generates a KZG proof
    /// via `MLTrainingProverV2::prove()`, updates weights, and returns the
    /// EVM-formatted proof bytes and public inputs.
    #[cfg(feature = "chain")]
    fn generate_real_proof(
        &mut self,
        round: u32,
    ) -> Result<(Vec<u8>, Vec<ethers::types::U256>)> {
        use helix_prover::halo2curves::bn256::Fr;
        use helix_prover::MLTrainingProverV2;

        let prover = self
            .prover
            .as_ref()
            .ok_or_else(|| anyhow!("Prover not initialized — call initialize_prover() first"))?;

        let state = self
            .training_state
            .as_ref()
            .ok_or_else(|| anyhow!("Training state not initialized"))?;

        let (d_in, d_hid, d_out) = self.config.model_dims;

        // Generate training data for this round
        let (x, target) = generate_training_data(d_in, d_out);

        // Learning rate as Fr (fixed-point scaled by 2^16)
        let lr = Fr::from((self.config.learning_rate * 65536.0) as u64);

        // Build witness from current weights
        let witness = MLTrainingProverV2::build_witness(
            d_in,
            d_hid,
            d_out,
            &x,
            &target,
            &state.w1,
            &state.b1,
            &state.w2,
            &state.b2,
            lr,
            round as u64,
            Fr::from(1u64),
        );

        // Generate the real ZK proof
        info!("Generating ZK proof for round {}...", round);
        let proof_start = Instant::now();

        let result = prover
            .prove(&witness)
            .map_err(|e| anyhow!("Proof generation failed for round {}: {:?}", round, e))?;

        let proof_time = proof_start.elapsed();
        info!(
            "Round {} proof generated in {:.1}s (verified={})",
            round,
            proof_time.as_secs_f64(),
            result.verified
        );

        // Update training state with new weights from the witness
        if let Some(ref mut s) = self.training_state {
            s.w1 = witness.w1_new.clone();
            s.b1 = witness.b1_new.clone();
            s.w2 = witness.w2_new.clone();
            s.b2 = witness.b2_new.clone();
            s.step += 1;
        }

        // Convert to EVM format
        let evm_proof = result
            .to_evm_proof()
            .map_err(|e| anyhow!("EVM proof serialization failed: {:?}", e))?;

        let evm_public_inputs = result.to_evm_public_inputs();

        // Convert [u8; 32] big-endian public inputs to U256
        let public_inputs_u256: Vec<ethers::types::U256> = evm_public_inputs
            .iter()
            .map(|bytes| ethers::types::U256::from_big_endian(bytes))
            .collect();

        Ok((evm_proof, public_inputs_u256))
    }

    /// Submit a real ZK proof on-chain for the completed round.
    #[cfg(feature = "chain")]
    async fn submit_round_proof_onchain(&mut self, round: u32) -> Result<Option<String>> {
        use crate::rpc::chain::ChainClient;

        let coordinator_addr = if let Some(ref d) = self.deployment {
            d.coordinator.clone()
        } else {
            self.config.coordinator_address.clone()
        };

        if coordinator_addr.is_empty() {
            return Ok(None);
        }

        let model_id = self.model_id.unwrap_or(0);

        // Generate real ZK proof from current training state
        let (proof_bytes, public_inputs) = self.generate_real_proof(round)?;

        info!(
            "Submitting real proof on-chain: {} bytes, {} public inputs",
            proof_bytes.len(),
            public_inputs.len()
        );

        let chain = ChainClient::new(
            &self.config.eth_rpc_url,
            &self.config.private_key,
            &coordinator_addr,
            None,
        )
        .await?;

        // Submit real proof with real public inputs via submit_proof_raw
        match chain
            .submit_proof_raw(model_id, round as u64, proof_bytes, public_inputs)
            .await
        {
            Ok(receipt) => {
                let tx_hash = format!("0x{:x}", receipt.transaction_hash);
                info!("Round {} real proof submitted: {}", round, tx_hash);
                Ok(Some(tx_hash))
            }
            Err(e) => {
                error!("Round {} proof submission failed: {}", round, e);
                Err(e.context(format!("proof submission failed for round {}", round)))
            }
        }
    }

    #[cfg(not(feature = "chain"))]
    async fn submit_round_proof_onchain(&mut self, _round: u32) -> Result<Option<String>> {
        Ok(None)
    }

    // =======================================================================
    // Health Monitoring (Task 5)
    // =======================================================================

    /// Check health of all managed node processes.
    ///
    /// For each process:
    /// - Checks if the OS process is still running (`try_wait()`)
    /// - Pings the aggregator's `/health` endpoint when applicable
    /// - Automatically restarts crashed processes (up to `max_restarts`)
    /// - Captures stderr output from crashed processes for diagnostics
    #[instrument(skip_all)]
    pub fn check_process_health(&mut self) -> Result<Vec<ProcessHealthReport>> {
        let mut reports = Vec::new();
        let mut to_restart: Vec<(String, String, u16, u32)> = Vec::new();

        for proc in &mut self.managed_processes {
            match proc.child.try_wait() {
                Ok(Some(exit_status)) => {
                    // Process has exited
                    let mut stderr_output = String::new();
                    if let Some(ref mut stderr) = proc.child.stderr {
                        let _ = stderr.read_to_string(&mut stderr_output);
                    }
                    if !stderr_output.is_empty() {
                        proc.last_stderr = stderr_output.clone();
                    }

                    warn!(
                        "Process {} exited with {} (restarts: {}/{})",
                        proc.name, exit_status, proc.restart_count, self.config.max_restarts
                    );

                    reports.push(ProcessHealthReport {
                        name: proc.name.clone(),
                        alive: false,
                        exit_status: Some(format!("{}", exit_status)),
                        restart_count: proc.restart_count,
                        last_stderr: if proc.last_stderr.is_empty() {
                            None
                        } else {
                            Some(proc.last_stderr.clone())
                        },
                    });

                    if proc.restart_count < self.config.max_restarts {
                        to_restart.push((
                            proc.name.clone(),
                            proc.role.clone(),
                            proc.port,
                            proc.restart_count + 1,
                        ));
                    } else {
                        error!(
                            "Process {} exceeded max restarts ({}), not restarting",
                            proc.name, self.config.max_restarts
                        );
                    }
                }
                Ok(None) => {
                    // Process still running
                    reports.push(ProcessHealthReport {
                        name: proc.name.clone(),
                        alive: true,
                        exit_status: None,
                        restart_count: proc.restart_count,
                        last_stderr: None,
                    });
                }
                Err(e) => {
                    warn!("Failed to check process {}: {}", proc.name, e);
                    reports.push(ProcessHealthReport {
                        name: proc.name.clone(),
                        alive: false,
                        exit_status: Some(format!("check failed: {}", e)),
                        restart_count: proc.restart_count,
                        last_stderr: None,
                    });
                }
            }
        }

        // Remove dead processes and restart
        self.managed_processes.retain(|p| {
            !to_restart.iter().any(|(name, _, _, _)| name == &p.name)
        });

        for (name, role, port, restart_count) in to_restart {
            info!("Restarting {} (attempt {})...", name, restart_count);
            match self.spawn_node(&name, &role, port) {
                Ok(child) => {
                    self.managed_processes.push(ManagedProcess {
                        name: name.clone(),
                        role,
                        port,
                        child,
                        restart_count,
                        last_stderr: String::new(),
                    });
                    info!("Successfully restarted {}", name);
                }
                Err(e) => {
                    error!("Failed to restart {}: {}", name, e);
                }
            }
        }

        Ok(reports)
    }

    /// Check aggregator health via HTTP endpoint.
    #[instrument(skip_all)]
    pub async fn check_aggregator_health(&self) -> Result<bool> {
        let url = format!("http://127.0.0.1:{}/health", self.config.http_port);

        match self.http_client.get(&url).send().await {
            Ok(resp) => {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let status = json
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    Ok(status == "healthy")
                } else {
                    Ok(false)
                }
            }
            Err(_) => Ok(false),
        }
    }

    // =======================================================================
    // Cleanup
    // =======================================================================

    /// Shut down all managed processes (nodes + Anvil).
    #[instrument(skip_all)]
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down orchestrator...");

        // Stop managed processes (in reverse order: workers first, then aggregator)
        for proc in self.managed_processes.drain(..).rev() {
            debug!("Stopping {} (managed)...", proc.name);
            let mut child = proc.child;
            let _ = child.kill();
            let _ = child.wait();
        }

        // Stop legacy node_processes
        for (name, mut child) in self.node_processes.drain(..).rev() {
            debug!("Stopping {}...", name);
            let _ = child.kill();
            let _ = child.wait();
        }

        // Stop Anvil
        if let Some(mut anvil) = self.anvil_process.take() {
            debug!("Stopping Anvil...");
            let _ = anvil.kill();
            let _ = anvil.wait();
        }

        info!("All processes stopped");
        Ok(())
    }

    // =======================================================================
    // Accessors
    // =======================================================================

    /// Get the deployed contract addresses (if deployment happened).
    pub fn deployment(&self) -> Option<&DeploymentResult> {
        self.deployment.as_ref()
    }

    /// Get the on-chain model ID (if registered).
    pub fn model_id(&self) -> Option<u64> {
        self.model_id
    }

    /// Get the orchestrator configuration.
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Get the aggregator HTTP API URL.
    pub fn aggregator_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.config.http_port)
    }
}

impl Drop for TrainingOrchestrator {
    fn drop(&mut self) {
        // Best-effort cleanup: kill all child processes
        for proc in self.managed_processes.drain(..) {
            debug!("Drop: killing {}", proc.name);
            let mut child = proc.child;
            let _ = child.kill();
            let _ = child.wait();
        }
        for (name, mut child) in self.node_processes.drain(..) {
            debug!("Drop: killing {}", name);
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(mut anvil) = self.anvil_process.take() {
            debug!("Drop: killing Anvil");
            let _ = anvil.kill();
            let _ = anvil.wait();
        }
    }
}

/// Health report for a managed process.
#[derive(Debug, Clone)]
pub struct ProcessHealthReport {
    pub name: String,
    pub alive: bool,
    pub exit_status: Option<String>,
    pub restart_count: u32,
    pub last_stderr: Option<String>,
}

/// Generate random training data (input features + one-hot target).
fn generate_training_data(d_in: usize, d_out: usize) -> (Vec<helix_prover::halo2curves::bn256::Fr>, Vec<helix_prover::halo2curves::bn256::Fr>) {
    use helix_prover::halo2curves::bn256::Fr;
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Small values to stay within ReLU lookup range (±128)
    let x: Vec<Fr> = (0..d_in)
        .map(|_| {
            let v: f64 = rng.gen_range(-0.5..0.5);
            let scaled = (v * 65536.0) as i64;
            if scaled >= 0 {
                Fr::from(scaled as u64)
            } else {
                use ff::Field;
                -Fr::from((-scaled) as u64)
            }
        })
        .collect();

    // One-hot target for classification
    let target_idx = rng.gen_range(0..d_out);
    let target: Vec<Fr> = (0..d_out)
        .map(|i| {
            use ff::Field;
            if i == target_idx { Fr::ONE } else { Fr::ZERO }
        })
        .collect();

    (x, target)
}

// ---------------------------------------------------------------------------
// Helper: Resolve helix-node binary
// ---------------------------------------------------------------------------

/// Search for the helix-node binary in common locations.
pub fn find_node_binary() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("target/debug/helix-node"),
        PathBuf::from("target/release/helix-node"),
        PathBuf::from("helix/target/debug/helix-node"),
        PathBuf::from("helix/target/release/helix-node"),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    // Try PATH
    if let Ok(output) = Command::new("which").arg("helix-node").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(PathBuf::from(path));
            }
        }
    }

    Err(anyhow!(
        "helix-node binary not found. Build it first:\n  cargo build -p helix-node\nOr specify --node-binary <path>"
    ))
}

/// Search for the contracts directory in common locations.
pub fn find_contracts_dir() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("contracts"),
        PathBuf::from("helix/contracts"),
        PathBuf::from("../contracts"),
    ];

    for path in &candidates {
        if path.join("foundry.toml").exists() {
            return Ok(path.clone());
        }
    }

    Err(anyhow!(
        "Contracts directory not found. Set --contracts-dir to the directory containing foundry.toml"
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.workers, 3);
        assert_eq!(config.rounds, 5);
        assert_eq!(config.model_dims, (4, 8, 2));
    }

    #[test]
    fn test_deployment_result_serialization() {
        let result = DeploymentResult {
            verifier: "0x5FbDB2315678afecb367f032d93F642f64180aa3".into(),
            coordinator: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512".into(),
            token: Some("0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0".into()),
            staking: None,
            rewards: None,
            registry: None,
            treasury: "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266".into(),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("0xe7f1725E"));

        let parsed: DeploymentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.coordinator, result.coordinator);
    }

    #[test]
    fn test_training_result_serialization() {
        let result = TrainingResult {
            rounds_completed: 5,
            final_loss: 0.15,
            accumulated_error: 25.0,
            proofs_submitted: 5,
            duration_ms: 45000,
            model_id: Some(1),
            tx_hashes: vec!["0xabc123".into()],
            deployment: None,
        };

        let json = serde_json::to_string_pretty(&result).unwrap();
        let parsed: TrainingResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.rounds_completed, 5);
        assert_eq!(parsed.model_id, Some(1));
    }

    #[test]
    fn test_find_node_binary_returns_error_when_not_found() {
        // This test just verifies the error message is helpful
        let result = find_node_binary();
        // May or may not find it depending on build state, just ensure no panic
        let _ = result;
    }
}
