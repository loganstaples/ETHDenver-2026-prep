//! Full owner-to-model orchestration for MPC-primary training.
//!
//! Single entry point for model owners: provide model config + worker endpoints,
//! and the orchestrator handles everything: chain deployment, job registration,
//! worker coordination, MPC training, checkpoint attestation, cheater slashing,
//! weight reconstruction, and accuracy verification.
//!
//! # Phases
//!
//! 1. **Data Loading** -- Load or generate MNIST training/test data
//! 2. **Weight Initialization** -- Xavier random init or load from file
//! 3. **Chain Bootstrap** -- Start local Anvil if no RPC URL provided
//! 4. **Contract Deployment** -- Deploy V4 coordinator if no address given
//! 5. **Job Registration** -- Register training job on-chain with payment
//! 6. **Worker Staking** -- Workers stake and join the training job
//! 7. **Worker Readiness** -- Poll on-chain until all workers have joined
//! 8. **MPC Training** -- Execute the full MPC training pipeline
//! 9. **Checkpoint Attestation** -- Submit each checkpoint to the contract
//! 10. **Cheater Slashing** -- Report MAC failures on-chain if detected
//! 11. **Training Completion** -- Finalize the job on-chain with signatures
//! 12. **Accuracy Evaluation** -- Evaluate model accuracy on held-out test set
//! 13. **Summary** -- Print comprehensive training report
//!
//! # Feature Gates
//!
//! All on-chain interactions require the `chain` feature. The MPC training
//! pipeline works without it, but phases 3-7, 9-11 are skipped.

use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

#[cfg(feature = "chain")]
use std::process::{Child, Command, Stdio};
#[cfg(feature = "chain")]
use std::str::FromStr;

#[cfg(feature = "chain")]
use ethers::prelude::*;
#[cfg(feature = "chain")]
use ethers::signers::{LocalWallet, Signer};
#[cfg(feature = "chain")]
use ethers::types::{Address, U256};
#[cfg(feature = "chain")]
use ethers::utils::keccak256;

use helix_mpc::e2e_integration::{
    CheaterRecord, FinalWeights, InitialWeights, MPCIntegrationConfig,
    MPCIntegrationResult,
};
use helix_mpc::mnist::{MnistDataset, MnistSample};

#[cfg(feature = "chain")]
use crate::rpc::chain_v4::{
    sign_checkpoint, sign_completion, sign_mac_failure, ChainClientV4,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a full owner-to-model orchestration run.
///
/// Specifies everything the model owner needs to provide: model architecture,
/// training hyperparameters, data source, worker endpoints, and on-chain config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullOrchestrationConfig {
    // -- Model Architecture --
    /// Layer dimensions, e.g. [784, 32, 10] for MNIST.
    /// Currently supports exactly 3 elements (input, hidden, output).
    pub architecture: Vec<usize>,

    // -- Training Hyperparameters --
    /// Total number of training steps.
    pub num_steps: usize,
    /// Learning rate for SGD.
    pub learning_rate: f64,
    /// How often to create Pedersen commitment checkpoints (every N steps).
    pub checkpoint_frequency: usize,
    /// How often to run MAC verification checks (every N steps, 0 to disable).
    pub mac_check_interval: u64,
    /// Number of Beaver triples to pre-generate per batch.
    pub beaver_batch_size: usize,
    /// Base random seed for deterministic execution.
    pub seed: u64,

    // -- Worker Configuration --
    /// TCP addresses of MPC workers, e.g. ["127.0.0.1:9001", "127.0.0.1:9002"].
    /// The number of entries determines the number of MPC parties.
    pub worker_endpoints: Vec<String>,

    // -- Weight Initialization --
    /// Path to a JSON file containing initial weights, or None for Xavier init.
    pub initial_weights_path: Option<String>,

    // -- Data Configuration --
    /// Whether to load real MNIST (requires network + `real-mnist` feature in helix-mpc).
    pub use_real_mnist: bool,
    /// Cache directory for downloaded MNIST files (None = default ~/.helix/data/mnist).
    pub mnist_cache_dir: Option<String>,
    /// Number of training samples to use.
    pub train_size: usize,
    /// Number of test samples to use (for accuracy evaluation).
    pub test_size: usize,

    // -- Chain Configuration (all behind #[cfg(feature = "chain")]) --
    /// Ethereum JSON-RPC URL. None = start a local Anvil instance.
    #[cfg(feature = "chain")]
    pub eth_rpc_url: Option<String>,
    /// Owner's Ethereum private key (hex, with or without 0x prefix).
    #[cfg(feature = "chain")]
    pub private_key: String,
    /// Private keys for demo-mode workers (owner controls all wallets).
    #[cfg(feature = "chain")]
    pub worker_private_keys: Vec<String>,
    /// ETH payment amount deposited with job registration.
    #[cfg(feature = "chain")]
    pub payment_amount_eth: f64,
    /// ETH stake amount each worker deposits when joining.
    #[cfg(feature = "chain")]
    pub stake_amount_eth: f64,
    /// Existing coordinator contract address. None = deploy a new V4.
    #[cfg(feature = "chain")]
    pub coordinator_address: Option<String>,
}

impl Default for FullOrchestrationConfig {
    fn default() -> Self {
        Self {
            architecture: vec![784, 32, 10],
            num_steps: 100,
            learning_rate: 0.001,
            checkpoint_frequency: 10,
            mac_check_interval: 10,
            beaver_batch_size: 2048,
            seed: 42,
            worker_endpoints: vec![
                "127.0.0.1:9001".to_string(),
                "127.0.0.1:9002".to_string(),
                "127.0.0.1:9003".to_string(),
            ],
            initial_weights_path: None,
            use_real_mnist: false,
            mnist_cache_dir: None,
            train_size: 1000,
            test_size: 200,
            #[cfg(feature = "chain")]
            eth_rpc_url: None,
            #[cfg(feature = "chain")]
            private_key: String::new(),
            #[cfg(feature = "chain")]
            worker_private_keys: Vec::new(),
            #[cfg(feature = "chain")]
            payment_amount_eth: 1.0,
            #[cfg(feature = "chain")]
            stake_amount_eth: 0.1,
            #[cfg(feature = "chain")]
            coordinator_address: None,
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Comprehensive result of a full orchestration run.
#[derive(Debug, Serialize, Deserialize)]
pub struct FullOrchestrationResult {
    /// On-chain job ID (0 if chain features disabled).
    pub job_id: u64,
    /// Total training steps completed.
    pub steps_completed: usize,
    /// Final loss value from the last training step.
    pub final_loss: f64,
    /// Per-step loss values.
    pub losses: Vec<f64>,
    /// Number of checkpoints successfully submitted on-chain.
    pub checkpoints_on_chain: usize,
    /// Number of MAC verification checks that passed.
    pub mac_checks_passed: usize,
    /// Cheater detection info, if any.
    pub cheater_detected: Option<CheaterInfo>,
    /// Final reconstructed model weights.
    #[serde(skip)]
    pub final_weights: Option<FinalWeights>,
    /// Test accuracy (0.0 to 1.0).
    pub test_accuracy: f64,
    /// Total training wall-clock time in seconds.
    pub training_time_secs: f64,
    /// Coordinator contract address (empty if chain features disabled).
    pub coordinator_address: String,
    /// Total gas used across all on-chain transactions.
    pub total_gas_used: u64,
}

/// Information about a detected cheater.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheaterInfo {
    /// The MPC party index of the cheater.
    pub party_index: usize,
    /// The training step at which cheating was detected.
    pub detected_at_step: u64,
    /// Whether the cheater was successfully slashed on-chain.
    pub slashed: bool,
    /// Transaction hash of the slash, if successful.
    pub slash_tx_hash: Option<String>,
}

// ============================================================================
// Orchestrator
// ============================================================================

/// Full lifecycle orchestrator for MPC-primary verifiable training.
///
/// Manages every phase from data loading through on-chain settlement:
///
/// ```text
/// Data -> Weights -> Anvil -> Deploy -> Register -> Stake -> Train -> Attest -> Slash -> Complete -> Eval
/// ```
///
/// All chain interactions are behind the `chain` feature gate. Without it,
/// the orchestrator still runs MPC training and evaluates accuracy.
pub struct FullOrchestrator {
    config: FullOrchestrationConfig,
    /// Anvil child process, if we started one.
    #[cfg(feature = "chain")]
    anvil_process: Option<Child>,
    /// RPC URL being used (either provided or Anvil's).
    #[cfg(feature = "chain")]
    rpc_url: Option<String>,
}

impl FullOrchestrator {
    /// Creates a new orchestrator with the given configuration.
    pub fn new(config: FullOrchestrationConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "chain")]
            anvil_process: None,
            #[cfg(feature = "chain")]
            rpc_url: None,
        }
    }

    /// Runs the full orchestration pipeline.
    ///
    /// Executes all 13 phases in sequence. On-chain phases are skipped
    /// if the `chain` feature is not enabled.
    pub async fn run(&mut self) -> Result<FullOrchestrationResult> {
        let overall_start = Instant::now();
        info!("Starting full orchestration pipeline");

        // Validate configuration up front.
        self.validate_config()
            .context("Configuration validation failed")?;

        let arch = &self.config.architecture;
        let d_in = arch[0];
        let d_hid = arch[1];
        let d_out = arch[2];
        let num_workers = self.config.worker_endpoints.len();

        info!(
            architecture = %format!("{}x{}x{}", d_in, d_hid, d_out),
            num_workers = num_workers,
            num_steps = self.config.num_steps,
            learning_rate = self.config.learning_rate,
            "Orchestration configuration validated"
        );

        // ================================================================
        // Phase 1: Load or generate training data
        // ================================================================
        info!("Phase 1: Loading training data");
        let phase1_start = Instant::now();

        let dataset = self
            .load_dataset(d_in, d_out)
            .context("Phase 1: Failed to load training data")?;

        let training_pairs = MnistDataset::as_training_pairs(&dataset.train);
        let test_samples = dataset.test.clone();

        info!(
            train_samples = training_pairs.len(),
            test_samples = test_samples.len(),
            elapsed_ms = phase1_start.elapsed().as_millis(),
            "Phase 1 complete: training data loaded"
        );

        // ================================================================
        // Phase 2: Generate or load initial weights
        // ================================================================
        info!("Phase 2: Initializing model weights");
        let phase2_start = Instant::now();

        let initial_weights = self
            .load_or_generate_weights(d_in, d_hid, d_out)
            .context("Phase 2: Failed to initialize weights")?;

        let total_params = d_hid * d_in + d_hid + d_out * d_hid + d_out;
        info!(
            total_params = total_params,
            source = if self.config.initial_weights_path.is_some() { "file" } else { "xavier" },
            elapsed_ms = phase2_start.elapsed().as_millis(),
            "Phase 2 complete: weights initialized"
        );

        // ================================================================
        // Phases 3-7: On-chain setup (feature-gated)
        // ================================================================
        #[cfg(feature = "chain")]
        let (job_id, coordinator_address, chain_client, worker_wallets, chain_gas) = {
            self.run_chain_setup_phases(d_in, d_hid, d_out, num_workers).await?
        };

        #[cfg(not(feature = "chain"))]
        let (job_id, coordinator_address, chain_gas): (u64, String, u64) =
            (0, String::new(), 0);

        // ================================================================
        // Phase 8: Run MPC training
        // ================================================================
        info!("Phase 8: Running MPC training");
        let phase8_start = Instant::now();

        let mpc_config = MPCIntegrationConfig {
            d_in,
            d_hid,
            d_out,
            num_workers,
            num_steps: self.config.num_steps,
            learning_rate: self.config.learning_rate,
            checkpoint_interval: self.config.checkpoint_frequency,
            mac_check_interval: self.config.mac_check_interval,
            beaver_batch_size: self.config.beaver_batch_size,
            initial_weights: Some(initial_weights),
            training_data: training_pairs,
            seed: self.config.seed,
            use_node_transport: false,
            use_tcp_transport: !self.config.worker_endpoints.is_empty(),
        };

        let mpc_result = helix_mpc::e2e_integration::run_mpc_training(mpc_config)
            .await
            .context("Phase 8: MPC training failed")?;

        info!(
            steps_completed = mpc_result.steps_completed,
            final_loss = mpc_result.final_loss,
            checkpoints = mpc_result.checkpoints.len(),
            mac_checks_passed = mpc_result.mac_checks_passed,
            cheater_detected = mpc_result.cheater_detected.is_some(),
            encrypted_distribution = mpc_result.encrypted_distribution,
            elapsed_ms = phase8_start.elapsed().as_millis(),
            "Phase 8 complete: MPC training finished"
        );

        // ================================================================
        // Phases 9-11: On-chain settlement (feature-gated)
        // ================================================================
        #[cfg(feature = "chain")]
        let (checkpoints_on_chain, cheater_info, settlement_gas) = {
            self.run_chain_settlement_phases(
                job_id,
                &mpc_result,
                &chain_client,
                &worker_wallets,
            )
            .await?
        };

        #[cfg(not(feature = "chain"))]
        let (checkpoints_on_chain, cheater_info, settlement_gas): (usize, Option<CheaterInfo>, u64) = {
            let ci = mpc_result.cheater_detected.as_ref().map(|c| CheaterInfo {
                party_index: c.party_index,
                detected_at_step: c.detected_at_step,
                slashed: false,
                slash_tx_hash: None,
            });
            (0, ci, 0)
        };

        // ================================================================
        // Phase 12: Evaluate accuracy on test set
        // ================================================================
        info!("Phase 12: Evaluating test accuracy");
        let phase12_start = Instant::now();

        let test_accuracy = evaluate_accuracy(&mpc_result.final_weights, &test_samples, d_in, d_hid, d_out);

        info!(
            test_accuracy = format!("{:.2}%", test_accuracy * 100.0),
            test_samples = test_samples.len(),
            elapsed_ms = phase12_start.elapsed().as_millis(),
            "Phase 12 complete: accuracy evaluated"
        );

        // ================================================================
        // Phase 13: Print comprehensive summary
        // ================================================================
        let total_elapsed = overall_start.elapsed().as_secs_f64();

        #[cfg(feature = "chain")]
        let total_gas = chain_gas + settlement_gas;
        #[cfg(not(feature = "chain"))]
        let total_gas = chain_gas + settlement_gas;

        let result = FullOrchestrationResult {
            job_id,
            steps_completed: mpc_result.steps_completed,
            final_loss: mpc_result.final_loss,
            losses: mpc_result.losses.clone(),
            checkpoints_on_chain,
            mac_checks_passed: mpc_result.mac_checks_passed,
            cheater_detected: cheater_info,
            final_weights: Some(mpc_result.final_weights),
            test_accuracy,
            training_time_secs: total_elapsed,
            coordinator_address: coordinator_address.clone(),
            total_gas_used: total_gas,
        };

        self.print_summary(&result);

        Ok(result)
    }

    /// Gracefully shuts down any spawned processes (Anvil).
    pub fn shutdown(&mut self) {
        #[cfg(feature = "chain")]
        {
            if let Some(mut child) = self.anvil_process.take() {
                info!("Shutting down Anvil process (pid: {})", child.id());
                match child.kill() {
                    Ok(()) => {
                        let _ = child.wait();
                        info!("Anvil process terminated");
                    }
                    Err(e) => {
                        warn!("Failed to kill Anvil process: {}", e);
                    }
                }
            }
        }
    }

    // ========================================================================
    // Validation
    // ========================================================================

    fn validate_config(&self) -> Result<()> {
        if self.config.architecture.len() != 3 {
            return Err(anyhow!(
                "Architecture must have exactly 3 elements [input, hidden, output], got {}",
                self.config.architecture.len()
            ));
        }

        for (i, &dim) in self.config.architecture.iter().enumerate() {
            if dim == 0 {
                return Err(anyhow!("Architecture dimension {} is zero", i));
            }
        }

        if self.config.worker_endpoints.is_empty() {
            return Err(anyhow!("At least one worker endpoint is required"));
        }

        if self.config.worker_endpoints.len() < 2 {
            return Err(anyhow!(
                "At least 2 workers are required for MPC, got {}",
                self.config.worker_endpoints.len()
            ));
        }

        if self.config.num_steps == 0 {
            return Err(anyhow!("num_steps must be > 0"));
        }

        if self.config.learning_rate <= 0.0 {
            return Err(anyhow!("learning_rate must be > 0.0"));
        }

        if self.config.checkpoint_frequency == 0 {
            return Err(anyhow!("checkpoint_frequency must be > 0"));
        }

        if self.config.train_size == 0 {
            return Err(anyhow!("train_size must be > 0"));
        }

        #[cfg(feature = "chain")]
        {
            if self.config.private_key.is_empty() {
                return Err(anyhow!("Owner private_key is required for on-chain interaction"));
            }
            if self.config.worker_private_keys.len() != self.config.worker_endpoints.len() {
                return Err(anyhow!(
                    "Number of worker_private_keys ({}) must match worker_endpoints ({})",
                    self.config.worker_private_keys.len(),
                    self.config.worker_endpoints.len()
                ));
            }
        }

        Ok(())
    }

    // ========================================================================
    // Phase 1: Data Loading
    // ========================================================================

    fn load_dataset(&self, _d_in: usize, _d_out: usize) -> Result<MnistDataset> {
        if self.config.use_real_mnist {
            // Real MNIST loading requires the `real-mnist` feature in helix-mpc.
            // Since helix-client depends on helix-mpc with default-features=false,
            // real MNIST is not available by default. To enable it, add
            // `features = ["real-mnist"]` to the helix-mpc dependency in Cargo.toml.
            //
            // For now, we fall back to synthetic data with a warning.
            warn!(
                "use_real_mnist=true but real-mnist feature is not enabled in helix-mpc; \
                 falling back to synthetic MNIST data. To enable real MNIST, add \
                 features = [\"real-mnist\"] to the helix-mpc dependency."
            );
            Ok(MnistDataset::generate(
                self.config.train_size,
                self.config.test_size,
                self.config.seed,
            ))
        } else {
            info!(
                train_size = self.config.train_size,
                test_size = self.config.test_size,
                seed = self.config.seed,
                "Generating synthetic MNIST data"
            );
            Ok(MnistDataset::generate(
                self.config.train_size,
                self.config.test_size,
                self.config.seed,
            ))
        }
    }

    // ========================================================================
    // Phase 2: Weight Initialization
    // ========================================================================

    fn load_or_generate_weights(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> Result<InitialWeights> {
        if let Some(ref path) = self.config.initial_weights_path {
            info!(path = %path, "Loading initial weights from file");
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read weights file: {}", path))?;
            let weights: InitialWeights = serde_json::from_str(&contents)
                .with_context(|| format!("Failed to parse weights JSON from: {}", path))?;

            // Validate dimensions.
            let expected_w1 = d_hid * d_in;
            if weights.w1.len() != expected_w1 {
                return Err(anyhow!(
                    "w1 length {} != expected {} ({}x{})",
                    weights.w1.len(),
                    expected_w1,
                    d_hid,
                    d_in
                ));
            }
            if weights.b1.len() != d_hid {
                return Err(anyhow!(
                    "b1 length {} != expected {}",
                    weights.b1.len(),
                    d_hid
                ));
            }
            let expected_w2 = d_out * d_hid;
            if weights.w2.len() != expected_w2 {
                return Err(anyhow!(
                    "w2 length {} != expected {} ({}x{})",
                    weights.w2.len(),
                    expected_w2,
                    d_out,
                    d_hid
                ));
            }
            if weights.b2.len() != d_out {
                return Err(anyhow!(
                    "b2 length {} != expected {}",
                    weights.b2.len(),
                    d_out
                ));
            }

            Ok(weights)
        } else {
            info!("Generating Xavier-initialized weights");
            Ok(xavier_init(d_in, d_hid, d_out, self.config.seed))
        }
    }

    // ========================================================================
    // Phases 3-7: On-chain setup (feature-gated)
    // ========================================================================

    /// Runs phases 3 through 7: Anvil startup, contract deployment, job
    /// registration, worker staking, and readiness polling.
    ///
    /// Returns (job_id, coordinator_address, chain_client, worker_wallets, gas_used).
    #[cfg(feature = "chain")]
    async fn run_chain_setup_phases(
        &mut self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        num_workers: usize,
    ) -> Result<(u64, String, ChainClientV4, Vec<LocalWallet>, u64)> {
        let mut total_gas: u64 = 0;

        // -- Phase 3: Start Anvil if needed --
        let rpc_url = if let Some(ref url) = self.config.eth_rpc_url {
            info!(rpc_url = %url, "Phase 3: Using provided RPC URL");
            url.clone()
        } else {
            info!("Phase 3: Starting local Anvil instance");
            let phase3_start = Instant::now();
            let url = self.start_anvil().context("Phase 3: Failed to start Anvil")?;
            info!(
                rpc_url = %url,
                elapsed_ms = phase3_start.elapsed().as_millis(),
                "Phase 3 complete: Anvil running"
            );
            url
        };
        self.rpc_url = Some(rpc_url.clone());

        // -- Phase 4: Deploy V4 contract if needed --
        let phase4_start = Instant::now();
        let (chain_client, coordinator_addr_str) = if let Some(ref addr) = self.config.coordinator_address {
            info!(address = %addr, "Phase 4: Using existing V4 coordinator");
            let client = ChainClientV4::new(&rpc_url, &self.config.private_key, addr, None)
                .await
                .context("Phase 4: Failed to connect to existing coordinator")?;
            (client, addr.clone())
        } else {
            info!("Phase 4: Deploying new V4 coordinator contract");
            let owner_pk = self.config.private_key.strip_prefix("0x")
                .unwrap_or(&self.config.private_key);
            let owner_wallet = LocalWallet::from_str(owner_pk)
                .map_err(|e| anyhow!("Invalid owner private key: {}", e))?;
            let treasury = owner_wallet.address();
            let verifier = Address::zero(); // No ZK verifier for MPC-primary mode.

            let (client, deploy_result) = ChainClientV4::deploy(
                &rpc_url,
                &self.config.private_key,
                treasury,
                verifier,
                None,
            )
            .await
            .context("Phase 4: V4 contract deployment failed")?;

            info!(
                coordinator = %deploy_result.coordinator,
                treasury = %deploy_result.treasury,
                elapsed_ms = phase4_start.elapsed().as_millis(),
                "Phase 4 complete: V4 coordinator deployed"
            );

            (client, deploy_result.coordinator)
        };

        // -- Phase 5: Register training job --
        info!("Phase 5: Registering training job");
        let phase5_start = Instant::now();

        let arch_hash = compute_architecture_hash(d_in, d_hid, d_out);
        let payment_wei = ethers::utils::parse_ether(self.config.payment_amount_eth)
            .context("Phase 5: Invalid payment amount")?;
        let num_checkpoints = self.config.num_steps / self.config.checkpoint_frequency;

        let (reg_receipt, job_id) = chain_client
            .register_training_job(
                arch_hash,
                self.config.checkpoint_frequency as u64,
                num_checkpoints.max(1) as u64,
                payment_wei,
            )
            .await
            .context("Phase 5: Job registration failed")?;

        let reg_gas = reg_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
        total_gas += reg_gas;

        info!(
            job_id = job_id,
            gas_used = reg_gas,
            elapsed_ms = phase5_start.elapsed().as_millis(),
            "Phase 5 complete: training job registered"
        );

        // -- Phase 6: Workers stake and join --
        info!("Phase 6: Workers staking and joining");
        let phase6_start = Instant::now();

        let stake_wei = ethers::utils::parse_ether(self.config.stake_amount_eth)
            .context("Phase 6: Invalid stake amount")?;

        let mut worker_wallets = Vec::with_capacity(num_workers);
        for (i, wk_key) in self.config.worker_private_keys.iter().enumerate() {
            let pk = wk_key.strip_prefix("0x").unwrap_or(wk_key);
            let provider = ethers::providers::Provider::<ethers::providers::Http>::try_from(&rpc_url)
                .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;
            let chain_id = provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64();
            let wallet = LocalWallet::from_str(pk)
                .map_err(|e| anyhow!("Invalid worker {} private key: {}", i, e))?
                .with_chain_id(chain_id);

            // In demo mode on Anvil, workers are funded by default accounts.
            // Create a per-worker client to send the staking transaction.
            let worker_client = ChainClientV4::with_wallet(
                &rpc_url,
                wallet.clone(),
                &coordinator_addr_str,
            )
            .await
            .with_context(|| format!("Phase 6: Failed to create worker {} client", i))?;

            let stake_receipt = worker_client
                .stake_and_join(job_id, stake_wei)
                .await
                .with_context(|| format!("Phase 6: Worker {} stake_and_join failed", i))?;

            let stake_gas = stake_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += stake_gas;

            info!(
                worker = i,
                address = %wallet.address(),
                gas_used = stake_gas,
                "Worker staked and joined"
            );

            worker_wallets.push(wallet);
        }

        info!(
            workers_staked = worker_wallets.len(),
            elapsed_ms = phase6_start.elapsed().as_millis(),
            "Phase 6 complete: all workers staked"
        );

        // -- Phase 7: Poll for worker readiness --
        info!("Phase 7: Waiting for all workers to be registered on-chain");
        let phase7_start = Instant::now();
        let expected_workers = num_workers as u64;
        let poll_timeout = Duration::from_secs(30);
        let poll_interval = Duration::from_millis(500);
        let deadline = Instant::now() + poll_timeout;

        loop {
            let count = chain_client
                .get_active_worker_count(job_id)
                .await
                .context("Phase 7: Failed to query worker count")?;

            debug!(active_workers = count, expected = expected_workers, "Polling worker count");

            if count >= expected_workers {
                info!(
                    active_workers = count,
                    elapsed_ms = phase7_start.elapsed().as_millis(),
                    "Phase 7 complete: all workers registered"
                );
                break;
            }

            if Instant::now() > deadline {
                return Err(anyhow!(
                    "Phase 7: Timed out waiting for workers. Expected {}, got {}",
                    expected_workers,
                    count
                ));
            }

            tokio::time::sleep(poll_interval).await;
        }

        Ok((job_id, coordinator_addr_str, chain_client, worker_wallets, total_gas))
    }

    // ========================================================================
    // Phases 9-11: On-chain settlement (feature-gated)
    // ========================================================================

    /// Runs phases 9 through 11: checkpoint attestation, cheater slashing,
    /// and training completion.
    ///
    /// Returns (checkpoints_on_chain, cheater_info, gas_used).
    #[cfg(feature = "chain")]
    async fn run_chain_settlement_phases(
        &self,
        job_id: u64,
        mpc_result: &MPCIntegrationResult,
        chain_client: &ChainClientV4,
        worker_wallets: &[LocalWallet],
    ) -> Result<(usize, Option<CheaterInfo>, u64)> {
        let mut total_gas: u64 = 0;
        let mut checkpoints_on_chain: usize = 0;

        // -- Phase 9: Submit checkpoints --
        info!(
            checkpoint_count = mpc_result.checkpoints.len(),
            "Phase 9: Submitting checkpoint attestations"
        );
        let phase9_start = Instant::now();

        for (idx, checkpoint) in mpc_result.checkpoints.iter().enumerate() {
            let step_u256 = U256::from(checkpoint.step);
            let loss_scaled = (checkpoint.loss * 1_000_000.0) as u64;
            let loss_u256 = U256::from(loss_scaled);

            // Collect signatures from all workers.
            let mut signatures = Vec::with_capacity(worker_wallets.len());
            for wallet in worker_wallets.iter() {
                let sig = sign_checkpoint(
                    wallet,
                    U256::from(job_id),
                    step_u256,
                    checkpoint.commitment_bytes32,
                    loss_u256,
                )
                .await
                .with_context(|| {
                    format!(
                        "Phase 9: Failed to sign checkpoint {} for worker {:?}",
                        idx,
                        wallet.address()
                    )
                })?;
                signatures.push(sig);
            }

            let receipt = chain_client
                .submit_checkpoint(
                    job_id,
                    checkpoint.step as u64,
                    checkpoint.commitment_bytes32,
                    loss_u256,
                    signatures,
                )
                .await
                .with_context(|| format!("Phase 9: Failed to submit checkpoint {}", idx))?;

            let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += gas;
            checkpoints_on_chain += 1;

            debug!(
                checkpoint_index = idx,
                step = checkpoint.step,
                loss = checkpoint.loss,
                gas_used = gas,
                "Checkpoint submitted on-chain"
            );
        }

        info!(
            checkpoints_submitted = checkpoints_on_chain,
            elapsed_ms = phase9_start.elapsed().as_millis(),
            "Phase 9 complete: all checkpoints attested"
        );

        // -- Phase 10: Report cheater if detected --
        let cheater_info = if let Some(ref cheater) = mpc_result.cheater_detected {
            info!(
                party_index = cheater.party_index,
                step = cheater.detected_at_step,
                "Phase 10: Reporting MAC failure on-chain"
            );
            let phase10_start = Instant::now();

            // Serialize failure evidence for on-chain reporting.
            let evidence_bytes = serialize_mac_evidence(cheater);

            // Determine the cheater's Ethereum address.
            let cheater_address = if cheater.party_index < worker_wallets.len() {
                worker_wallets[cheater.party_index].address()
            } else {
                warn!(
                    "Cheater party index {} exceeds wallet count {}, using zero address",
                    cheater.party_index,
                    worker_wallets.len()
                );
                Address::zero()
            };

            // Collect signatures from honest workers (all except the cheater).
            let mut reporter_sigs = Vec::new();
            for (i, wallet) in worker_wallets.iter().enumerate() {
                if i == cheater.party_index {
                    continue; // Skip the cheater.
                }
                let sig = sign_mac_failure(
                    wallet,
                    U256::from(job_id),
                    U256::from(cheater.detected_at_step),
                    cheater_address,
                    &evidence_bytes,
                )
                .await
                .with_context(|| {
                    format!("Phase 10: Worker {} failed to sign MAC failure report", i)
                })?;
                reporter_sigs.push(sig);
            }

            let slash_result = chain_client
                .report_mac_failure(
                    job_id,
                    cheater.detected_at_step,
                    cheater_address,
                    evidence_bytes,
                    reporter_sigs,
                )
                .await;

            let (slashed, slash_tx_hash) = match slash_result {
                Ok(receipt) => {
                    let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
                    total_gas += gas;
                    let tx_hash = format!("{:?}", receipt.transaction_hash);
                    info!(
                        tx_hash = %tx_hash,
                        gas_used = gas,
                        elapsed_ms = phase10_start.elapsed().as_millis(),
                        "Phase 10 complete: cheater slashed"
                    );
                    (true, Some(tx_hash))
                }
                Err(e) => {
                    error!("Phase 10: MAC failure report failed: {}", e);
                    (false, None)
                }
            };

            Some(CheaterInfo {
                party_index: cheater.party_index,
                detected_at_step: cheater.detected_at_step,
                slashed,
                slash_tx_hash,
            })
        } else {
            info!("Phase 10: No cheater detected, skipping");
            None
        };

        // -- Phase 11: Complete training on-chain --
        info!("Phase 11: Completing training on-chain");
        let phase11_start = Instant::now();

        // Use the last checkpoint commitment as the final commitment, or compute one.
        let final_commitment = if let Some(last_cp) = mpc_result.checkpoints.last() {
            last_cp.commitment_bytes32
        } else {
            // No checkpoints: compute a commitment from the final weights.
            compute_weight_commitment(&mpc_result.final_weights)
        };

        // Collect completion signatures from all honest workers.
        let honest_wallets: Vec<&LocalWallet> = if let Some(ref ci) = cheater_info {
            worker_wallets
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ci.party_index)
                .map(|(_, w)| w)
                .collect()
        } else {
            worker_wallets.iter().collect()
        };

        let mut completion_sigs = Vec::with_capacity(honest_wallets.len());
        for wallet in &honest_wallets {
            let sig = sign_completion(wallet, U256::from(job_id), final_commitment)
                .await
                .context("Phase 11: Failed to sign completion message")?;
            completion_sigs.push(sig);
        }

        let completion_receipt = chain_client
            .complete_training(job_id, final_commitment, completion_sigs)
            .await
            .context("Phase 11: complete_training failed")?;

        let completion_gas = completion_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
        total_gas += completion_gas;

        info!(
            tx_hash = %format!("{:?}", completion_receipt.transaction_hash),
            gas_used = completion_gas,
            elapsed_ms = phase11_start.elapsed().as_millis(),
            "Phase 11 complete: training finalized on-chain"
        );

        // Verify final job state.
        let job_summary = chain_client
            .get_job_summary(job_id)
            .await
            .context("Failed to query final job summary")?;

        info!(
            job_completed = job_summary.completed,
            final_step = job_summary.current_step,
            active_workers = job_summary.active_worker_count,
            "On-chain job final state"
        );

        Ok((checkpoints_on_chain, cheater_info, total_gas))
    }

    // ========================================================================
    // Anvil Management
    // ========================================================================

    /// Starts a local Anvil instance and returns its RPC URL.
    ///
    /// Spawns `anvil` as a child process, then polls by attempting TCP
    /// connections until the node is ready (up to 15 seconds).
    #[cfg(feature = "chain")]
    fn start_anvil(&mut self) -> Result<String> {
        let port = 8545u16;
        let rpc_url = format!("http://127.0.0.1:{}", port);

        info!(port = port, "Spawning Anvil process");

        let child = Command::new("anvil")
            .args([
                "--port",
                &port.to_string(),
                "--accounts",
                "20",
                "--balance",
                "10000",
                "--silent",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn Anvil. Is Foundry installed?")?;

        self.anvil_process = Some(child);

        // Poll for readiness by attempting TCP connections to the Anvil port.
        // Once TCP connects, Anvil's HTTP server is ready.
        let deadline = Instant::now() + Duration::from_secs(15);
        let addr: std::net::SocketAddr = format!("127.0.0.1:{}", port)
            .parse()
            .context("Failed to parse Anvil socket address")?;

        loop {
            match std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
                Ok(_stream) => {
                    info!(rpc_url = %rpc_url, "Anvil is ready (TCP connection succeeded)");
                    return Ok(rpc_url);
                }
                Err(_) => {
                    debug!("Anvil not responding yet, retrying...");
                }
            }

            if Instant::now() > deadline {
                // Kill the process if it never became ready.
                self.shutdown();
                return Err(anyhow!(
                    "Anvil failed to become ready within 15 seconds at {}",
                    rpc_url
                ));
            }

            std::thread::sleep(Duration::from_millis(200));
        }
    }

    // ========================================================================
    // Summary Printing
    // ========================================================================

    fn print_summary(&self, result: &FullOrchestrationResult) {
        let arch = &self.config.architecture;
        let d_in = arch[0];
        let d_hid = arch[1];
        let d_out = arch[2];
        let total_params = d_hid * d_in + d_hid + d_out * d_hid + d_out;

        let divider = "=".repeat(64);
        let thin_divider = "-".repeat(64);

        info!("\n{divider}");
        info!("  HELIX Full Orchestration Summary");
        info!("{divider}");
        info!("");
        info!("  Model Architecture");
        info!("  {thin_divider}");
        info!("    Layers:           {}x{}x{}", d_in, d_hid, d_out);
        info!("    Parameters:       {}", total_params);
        info!("    W1: {}x{} = {}", d_hid, d_in, d_hid * d_in);
        info!("    b1: {}", d_hid);
        info!("    W2: {}x{} = {}", d_out, d_hid, d_out * d_hid);
        info!("    b2: {}", d_out);
        info!("");
        info!("  Training Results");
        info!("  {thin_divider}");
        info!("    Steps:            {}/{}", result.steps_completed, self.config.num_steps);
        info!("    Final Loss:       {:.6}", result.final_loss);
        info!("    Test Accuracy:    {:.2}%", result.test_accuracy * 100.0);
        info!("    Workers:          {}", self.config.worker_endpoints.len());
        info!("    MAC Checks:       {} passed", result.mac_checks_passed);
        info!("    Time:             {:.2}s", result.training_time_secs);
        info!("");
        info!("  Loss Progression");
        info!("  {thin_divider}");

        let loss_count = result.losses.len();
        if loss_count > 0 {
            // Show first, middle, and last loss values.
            let step_indices = if loss_count <= 10 {
                (0..loss_count).collect::<Vec<_>>()
            } else {
                let mut indices = Vec::new();
                for i in 0..5 {
                    indices.push(i);
                }
                indices.push(loss_count / 2);
                for i in (loss_count - 4)..loss_count {
                    indices.push(i);
                }
                indices
            };

            for &i in &step_indices {
                info!("    Step {:>4}: loss = {:.6}", i + 1, result.losses[i]);
            }
        }

        info!("");
        info!("  On-Chain Settlement");
        info!("  {thin_divider}");

        if !result.coordinator_address.is_empty() {
            info!("    Job ID:           {}", result.job_id);
            info!("    Coordinator:      {}", result.coordinator_address);
            info!("    Checkpoints:      {} on-chain", result.checkpoints_on_chain);
            info!("    Total Gas:        {}", result.total_gas_used);
        } else {
            info!("    (chain features not enabled)");
        }

        if let Some(ref cheater) = result.cheater_detected {
            info!("");
            info!("  Cheater Detection");
            info!("  {thin_divider}");
            info!("    Party:            {}", cheater.party_index);
            info!("    Detected at Step: {}", cheater.detected_at_step);
            info!("    Slashed:          {}", cheater.slashed);
            if let Some(ref tx) = cheater.slash_tx_hash {
                info!("    Slash TX:         {}", tx);
            }
        } else {
            info!("");
            info!("  Integrity: All MAC checks passed, no cheating detected");
        }

        info!("");
        info!("{divider}");
    }
}

impl Drop for FullOrchestrator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Xavier (Glorot) initialization for a 2-layer MLP.
///
/// Generates random weights using the standard Xavier uniform distribution:
///   W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
/// Biases are initialized to zero.
pub fn xavier_init(d_in: usize, d_hid: usize, d_out: usize, seed: u64) -> InitialWeights {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // W1: d_hid x d_in
    let limit1 = (6.0_f64 / (d_in + d_hid) as f64).sqrt();
    let w1: Vec<f64> = (0..d_hid * d_in)
        .map(|_| rng.gen_range(-limit1..limit1))
        .collect();
    let b1 = vec![0.0; d_hid];

    // W2: d_out x d_hid
    let limit2 = (6.0_f64 / (d_hid + d_out) as f64).sqrt();
    let w2: Vec<f64> = (0..d_out * d_hid)
        .map(|_| rng.gen_range(-limit2..limit2))
        .collect();
    let b2 = vec![0.0; d_out];

    InitialWeights { w1, b1, w2, b2 }
}

/// Evaluates classification accuracy on a test set using a 2-layer MLP.
///
/// Forward pass: output = softmax(W2 * relu(W1 * x + b1) + b2)
/// Accuracy = fraction of samples where argmax(output) == argmax(label)
pub fn evaluate_accuracy(
    weights: &FinalWeights,
    test_samples: &[MnistSample],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> f64 {
    if test_samples.is_empty() {
        warn!("No test samples provided for accuracy evaluation");
        return 0.0;
    }

    let mut correct = 0usize;

    for sample in test_samples {
        // Layer 1: hidden = relu(W1 * x + b1)
        // W1 is stored as d_hid x d_in (row-major).
        let mut hidden = vec![0.0f64; d_hid];
        for h in 0..d_hid {
            let mut sum = weights.b1[h];
            for j in 0..d_in {
                sum += weights.w1[h * d_in + j] * sample.pixels[j];
            }
            // ReLU activation.
            hidden[h] = sum.max(0.0);
        }

        // Layer 2: output = W2 * hidden + b2
        // W2 is stored as d_out x d_hid (row-major).
        let mut output = vec![0.0f64; d_out];
        for o in 0..d_out {
            let mut sum = weights.b2[o];
            for h in 0..d_hid {
                sum += weights.w2[o * d_hid + h] * hidden[h];
            }
            output[o] = sum;
        }

        // Argmax of output.
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Argmax of label (ground truth).
        let actual = sample
            .label
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if predicted == actual {
            correct += 1;
        }
    }

    correct as f64 / test_samples.len() as f64
}

/// Computes a keccak256 hash of the model architecture for on-chain registration.
#[cfg(feature = "chain")]
fn compute_architecture_hash(d_in: usize, d_hid: usize, d_out: usize) -> [u8; 32] {
    let arch_string = format!("HELIX_ARCH:{}:{}:{}", d_in, d_hid, d_out);
    keccak256(arch_string.as_bytes())
}

/// Computes a weight commitment hash from final weights.
///
/// Used as a fallback final commitment when no checkpoints were produced.
/// Hashes the concatenation of all weight bytes using keccak256.
#[cfg(feature = "chain")]
fn compute_weight_commitment(weights: &FinalWeights) -> [u8; 32] {
    let mut data = Vec::new();
    for &v in weights.w1.iter().chain(weights.b1.iter()).chain(weights.w2.iter()).chain(weights.b2.iter()) {
        data.extend_from_slice(&v.to_le_bytes());
    }
    keccak256(&data)
}

/// Serializes a CheaterRecord's failure evidence into bytes for on-chain reporting.
///
/// The evidence format is:
///   [4 bytes: party_index as u32] || [8 bytes: step as u64] || [serialized sigma values]
#[cfg(feature = "chain")]
fn serialize_mac_evidence(cheater: &CheaterRecord) -> Vec<u8> {
    let mut evidence = Vec::new();

    // Party index (4 bytes, big-endian).
    evidence.extend_from_slice(&(cheater.party_index as u32).to_be_bytes());

    // Step number (8 bytes, big-endian).
    evidence.extend_from_slice(&cheater.detected_at_step.to_be_bytes());

    // Sigma values from the failure report.
    for sigma in &cheater.failure_report.sigma_values {
        // Length-prefix each sigma value.
        evidence.extend_from_slice(&(sigma.len() as u32).to_be_bytes());
        evidence.extend_from_slice(sigma);
    }

    // Commitments from the failure report.
    for commitment in &cheater.failure_report.commitments {
        evidence.extend_from_slice(commitment);
    }

    // Pairwise check results summary.
    let pairwise = &cheater.failure_report.evidence.pairwise_results;
    evidence.extend_from_slice(&(pairwise.len() as u32).to_be_bytes());
    for result in pairwise {
        evidence.extend_from_slice(&(result.party_a as u32).to_be_bytes());
        evidence.extend_from_slice(&(result.party_b as u32).to_be_bytes());
    }

    evidence
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_init_dimensions() {
        let weights = xavier_init(784, 32, 10, 42);
        assert_eq!(weights.w1.len(), 32 * 784);
        assert_eq!(weights.b1.len(), 32);
        assert_eq!(weights.w2.len(), 10 * 32);
        assert_eq!(weights.b2.len(), 10);
    }

    #[test]
    fn test_xavier_init_range() {
        let weights = xavier_init(784, 32, 10, 42);
        let limit1 = (6.0_f64 / (784.0 + 32.0)).sqrt();
        for &w in &weights.w1 {
            assert!(w >= -limit1 && w <= limit1, "w1 value {} out of range", w);
        }
        let limit2 = (6.0_f64 / (32.0 + 10.0)).sqrt();
        for &w in &weights.w2 {
            assert!(w >= -limit2 && w <= limit2, "w2 value {} out of range", w);
        }
    }

    #[test]
    fn test_xavier_init_biases_zero() {
        let weights = xavier_init(784, 32, 10, 42);
        for &b in &weights.b1 {
            assert_eq!(b, 0.0);
        }
        for &b in &weights.b2 {
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn test_xavier_init_deterministic() {
        let w1 = xavier_init(784, 32, 10, 42);
        let w2 = xavier_init(784, 32, 10, 42);
        assert_eq!(w1.w1, w2.w1);
        assert_eq!(w1.w2, w2.w2);
    }

    #[test]
    fn test_xavier_init_different_seeds() {
        let w1 = xavier_init(784, 32, 10, 42);
        let w2 = xavier_init(784, 32, 10, 99);
        assert_ne!(w1.w1, w2.w1);
    }

    #[test]
    fn test_evaluate_accuracy_perfect() {
        // Create a trivially solvable scenario: 2 classes with identity-like weights.
        let weights = FinalWeights {
            w1: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b2: vec![0.0, 0.0],
        };

        let samples = vec![
            MnistSample {
                pixels: vec![1.0, 0.0],
                label: vec![1.0, 0.0],
                digit: 0,
            },
            MnistSample {
                pixels: vec![0.0, 1.0],
                label: vec![0.0, 1.0],
                digit: 1,
            },
        ];

        let acc = evaluate_accuracy(&weights, &samples, 2, 2, 2);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_evaluate_accuracy_empty() {
        let weights = FinalWeights {
            w1: vec![],
            b1: vec![],
            w2: vec![],
            b2: vec![],
        };
        let acc = evaluate_accuracy(&weights, &[], 0, 0, 0);
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn test_evaluate_accuracy_random_baseline() {
        // Random weights should give roughly 50% on a balanced binary task,
        // but we just check it returns a valid value.
        let weights = xavier_init(2, 4, 2, 42);
        let fw = FinalWeights {
            w1: weights.w1,
            b1: weights.b1,
            w2: weights.w2,
            b2: weights.b2,
        };
        let samples = vec![
            MnistSample {
                pixels: vec![1.0, 0.0],
                label: vec![1.0, 0.0],
                digit: 0,
            },
            MnistSample {
                pixels: vec![0.0, 1.0],
                label: vec![0.0, 1.0],
                digit: 1,
            },
        ];
        let acc = evaluate_accuracy(&fw, &samples, 2, 4, 2);
        assert!(acc >= 0.0 && acc <= 1.0);
    }

    #[test]
    fn test_config_default_architecture() {
        let config = FullOrchestrationConfig::default();
        assert_eq!(config.architecture, vec![784, 32, 10]);
        assert_eq!(config.num_steps, 100);
        assert_eq!(config.worker_endpoints.len(), 3);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = FullOrchestrationConfig {
            architecture: vec![784, 32, 10],
            num_steps: 50,
            learning_rate: 0.01,
            checkpoint_frequency: 5,
            mac_check_interval: 5,
            beaver_batch_size: 512,
            seed: 42,
            worker_endpoints: vec![
                "127.0.0.1:9001".to_string(),
                "127.0.0.1:9002".to_string(),
                "127.0.0.1:9003".to_string(),
            ],
            initial_weights_path: None,
            use_real_mnist: false,
            mnist_cache_dir: None,
            train_size: 100,
            test_size: 20,
            #[cfg(feature = "chain")]
            eth_rpc_url: None,
            #[cfg(feature = "chain")]
            private_key: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".to_string(),
            #[cfg(feature = "chain")]
            worker_private_keys: vec![
                "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d".to_string(),
                "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a".to_string(),
                "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6".to_string(),
            ],
            #[cfg(feature = "chain")]
            payment_amount_eth: 1.0,
            #[cfg(feature = "chain")]
            stake_amount_eth: 0.1,
            #[cfg(feature = "chain")]
            coordinator_address: None,
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_ok());
    }

    #[test]
    fn test_config_validation_bad_architecture() {
        let config = FullOrchestrationConfig {
            architecture: vec![784, 32], // Only 2 elements
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_config_validation_zero_dim() {
        let config = FullOrchestrationConfig {
            architecture: vec![784, 0, 10],
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_config_validation_no_workers() {
        let config = FullOrchestrationConfig {
            worker_endpoints: vec![],
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_config_validation_one_worker() {
        let config = FullOrchestrationConfig {
            worker_endpoints: vec!["127.0.0.1:9001".to_string()],
            #[cfg(feature = "chain")]
            worker_private_keys: vec![
                "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d".to_string(),
            ],
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_config_validation_zero_steps() {
        let config = FullOrchestrationConfig {
            num_steps: 0,
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_config_validation_negative_lr() {
        let config = FullOrchestrationConfig {
            learning_rate: -0.01,
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        assert!(orchestrator.validate_config().is_err());
    }

    #[test]
    fn test_load_synthetic_data() {
        let config = FullOrchestrationConfig {
            use_real_mnist: false,
            train_size: 50,
            test_size: 10,
            seed: 42,
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        let dataset = orchestrator.load_dataset(784, 10).unwrap();
        assert_eq!(dataset.train.len(), 50);
        assert_eq!(dataset.test.len(), 10);
        assert_eq!(dataset.train[0].pixels.len(), 784);
        assert_eq!(dataset.train[0].label.len(), 10);
    }

    #[test]
    fn test_load_weights_xavier() {
        let config = FullOrchestrationConfig {
            initial_weights_path: None,
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        let weights = orchestrator.load_or_generate_weights(784, 32, 10).unwrap();
        assert_eq!(weights.w1.len(), 32 * 784);
        assert_eq!(weights.b1.len(), 32);
        assert_eq!(weights.w2.len(), 10 * 32);
        assert_eq!(weights.b2.len(), 10);
    }

    #[test]
    fn test_load_weights_from_file() {
        let weights = xavier_init(4, 2, 3, 42);
        let json = serde_json::to_string(&weights).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &json).unwrap();

        let config = FullOrchestrationConfig {
            initial_weights_path: Some(tmp.path().to_string_lossy().to_string()),
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        let loaded = orchestrator.load_or_generate_weights(4, 2, 3).unwrap();
        assert_eq!(loaded.w1.len(), weights.w1.len());
        assert_eq!(loaded.w2.len(), weights.w2.len());
        // Compare with tolerance due to JSON float serialization precision.
        for (a, b) in loaded.w1.iter().zip(weights.w1.iter()) {
            assert!((a - b).abs() < 1e-14, "w1 mismatch: {} vs {}", a, b);
        }
        for (a, b) in loaded.w2.iter().zip(weights.w2.iter()) {
            assert!((a - b).abs() < 1e-14, "w2 mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_load_weights_file_wrong_dims() {
        let weights = xavier_init(4, 2, 3, 42);
        let json = serde_json::to_string(&weights).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &json).unwrap();

        let config = FullOrchestrationConfig {
            initial_weights_path: Some(tmp.path().to_string_lossy().to_string()),
            ..FullOrchestrationConfig::default()
        };
        let orchestrator = FullOrchestrator::new(config);
        // Expect failure: weights are 4x2x3 but we ask for 784x32x10.
        assert!(orchestrator.load_or_generate_weights(784, 32, 10).is_err());
    }

    #[cfg(feature = "chain")]
    #[test]
    fn test_compute_architecture_hash_deterministic() {
        let h1 = compute_architecture_hash(784, 32, 10);
        let h2 = compute_architecture_hash(784, 32, 10);
        assert_eq!(h1, h2);
    }

    #[cfg(feature = "chain")]
    #[test]
    fn test_compute_architecture_hash_varies() {
        let h1 = compute_architecture_hash(784, 32, 10);
        let h2 = compute_architecture_hash(784, 64, 10);
        assert_ne!(h1, h2);
    }

    #[cfg(feature = "chain")]
    #[test]
    fn test_compute_weight_commitment() {
        let w = FinalWeights {
            w1: vec![1.0, 2.0],
            b1: vec![0.5],
            w2: vec![3.0],
            b2: vec![0.1],
        };
        let c1 = compute_weight_commitment(&w);
        let c2 = compute_weight_commitment(&w);
        assert_eq!(c1, c2);
        assert_eq!(c1.len(), 32);
    }

    #[test]
    fn test_orchestrator_drop_cleanup() {
        // Verify that Drop is implemented (no Anvil to clean up in this case).
        let config = FullOrchestrationConfig::default();
        let _orch = FullOrchestrator::new(config);
        // Drop happens here -- should not panic.
    }

    #[test]
    fn test_cheater_info_serialization() {
        let info = CheaterInfo {
            party_index: 2,
            detected_at_step: 15,
            slashed: true,
            slash_tx_hash: Some("0xabc123".to_string()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: CheaterInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.party_index, 2);
        assert_eq!(parsed.detected_at_step, 15);
        assert!(parsed.slashed);
        assert_eq!(parsed.slash_tx_hash, Some("0xabc123".to_string()));
    }

    #[test]
    fn test_result_serialization() {
        let result = FullOrchestrationResult {
            job_id: 1,
            steps_completed: 50,
            final_loss: 0.123,
            losses: vec![1.0, 0.5, 0.25, 0.123],
            checkpoints_on_chain: 5,
            mac_checks_passed: 10,
            cheater_detected: None,
            final_weights: None,
            test_accuracy: 0.95,
            training_time_secs: 12.5,
            coordinator_address: "0x1234".to_string(),
            total_gas_used: 500_000,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: FullOrchestrationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.job_id, 1);
        assert_eq!(parsed.steps_completed, 50);
        assert_eq!(parsed.test_accuracy, 0.95);
    }

    #[test]
    fn test_evaluate_accuracy_relu_behavior() {
        // Test that ReLU is correctly applied (negative values become 0).
        // W1 = [[-1, 0], [0, 1]], input = [1.0, 0.0]
        // hidden = relu([-1, 0]) = [0, 0]
        // With output bias b2 = [0, 0] and W2 = identity,
        // output = [0, 0] -> all equal, so argmax is ambiguous.
        //
        // Instead, set b2 to break the tie and verify ReLU effect.
        let weights = FinalWeights {
            w1: vec![-1.0, 0.0, 0.0, 1.0], // 2x2: first neuron inverts input[0]
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 0.0, 0.0, 1.0], // 2x2 identity
            b2: vec![0.5, 0.0],            // bias pushes class 0 up
        };

        let samples = vec![MnistSample {
            pixels: vec![1.0, 0.0],
            label: vec![0.0, 1.0], // class 1
            digit: 1,
        }];

        // hidden = relu([-1*1 + 0*0, 0*1 + 1*0]) = relu([-1, 0]) = [0, 0]
        // output = W2 * [0, 0] + [0.5, 0] = [0.5, 0.0]
        // argmax(output) = 0
        // argmax(label) = 1
        // Wrong prediction: accuracy = 0.
        let acc = evaluate_accuracy(&weights, &samples, 2, 2, 2);
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn test_evaluate_accuracy_with_bias() {
        // Test that biases are correctly applied.
        let weights = FinalWeights {
            w1: vec![0.0, 0.0, 0.0, 0.0], // zero weights
            b1: vec![1.0, 0.0],            // bias pushes first hidden neuron positive
            w2: vec![1.0, 0.0, 0.0, 1.0], // identity
            b2: vec![0.0, 0.0],
        };

        let samples = vec![MnistSample {
            pixels: vec![0.0, 0.0],
            label: vec![1.0, 0.0], // class 0
            digit: 0,
        }];

        // hidden = relu([1.0, 0.0]) = [1.0, 0.0]
        // output = [1.0, 0.0] -> argmax = 0
        // label argmax = 0
        // Correct.
        let acc = evaluate_accuracy(&weights, &samples, 2, 2, 2);
        assert_eq!(acc, 1.0);
    }
}
