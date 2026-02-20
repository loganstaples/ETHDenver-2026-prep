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

use std::sync::Arc;
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

use crate::zk_proof_layer::{ZkCheckpointProofResult, ZkProofConfig, ZkProofLayer};

// ============================================================================
// ZK Mode
// ============================================================================

/// Controls when ZK proofs are required for checkpoint submissions.
///
/// - `Off`: Pure MPC+MAC attestation, no ZK proofs.
/// - `Always`: ZK proofs are always required for every checkpoint (or per the configured frequency).
/// - `Risk { min_workers }`: Start with MPC-only, but auto-activate ZK proofs on-chain when
///   the active worker count drops below `min_workers` (e.g. due to cheater slashing).
///   This provides a safety net: if too many workers are removed, the remaining MPC quorum
///   may be too small for information-theoretic security, so ZK proofs fill the gap.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ZkMode {
    /// No ZK proofs — pure MPC+MAC attestation.
    Off,
    /// Always require ZK proofs for checkpoints.
    Always,
    /// Auto-activate ZK when active workers drop below the threshold.
    Risk {
        /// Minimum number of active workers for MPC-only mode.
        /// When active workers fall below this, ZK proofs are required.
        min_workers: usize,
    },
}

impl Default for ZkMode {
    fn default() -> Self {
        ZkMode::Off
    }
}

#[cfg(feature = "chain")]
use crate::rpc::chain_v4::{
    sign_checkpoint, sign_completion, sign_mac_failure, ChainClientV4, ModelStoreClient,
};

// ============================================================================
// Progress Reporting
// ============================================================================

/// Events emitted by the orchestrator for user-facing progress display.
///
/// The orchestrator calls the progress callback at key moments so the CLI
/// can render formatted output (colored text, progress bars, etc.) without
/// coupling the orchestrator to any particular display mechanism.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// A phase is starting. Fields: (phase_number, total_phases, description).
    PhaseStarted { phase: u32, total: u32, description: String },
    /// A phase completed successfully. Fields: (phase_number, elapsed_ms).
    PhaseCompleted { phase: u32, elapsed_ms: u128 },
    /// Training step progress. Fields: (step, total_steps, loss, accuracy_estimate, mac_ok).
    TrainingStep { step: usize, total: usize, loss: f64, accuracy: f64, mac_ok: bool },
    /// Sub-step progress within a single training step (e.g. forward pass, backward pass).
    SubStep { step: usize, total: usize, operation: String },
    /// A checkpoint was submitted on-chain.
    CheckpointSubmitted { index: usize, total: usize, step: u64, tx_hash: String },
    /// A cheater was detected.
    CheaterDetected { party_index: usize, step: u64 },
    /// Cheater was slashed on-chain.
    CheaterSlashed { party_index: usize, tx_hash: String },
    /// Training completed with summary stats.
    TrainingComplete { accuracy: f64, steps: usize, time_secs: f64, checkpoints: usize },
    /// An error occurred but was recovered from.
    RecoverableError { phase: u32, message: String, retry_count: u32 },
    /// ZK proof generation started for a checkpoint.
    ZkProofStarted { checkpoint_index: usize, step: u64 },
    /// ZK proof generation succeeded.
    ZkProofGenerated { checkpoint_index: usize, step: u64, proof_size: usize, time_ms: u64, verified: bool },
    /// ZK proof generation failed (non-fatal).
    ZkProofFailed { checkpoint_index: usize, step: u64, error: String },
    /// ZK proof submitted on-chain.
    ZkProofSubmitted { checkpoint_index: usize, step: u64, tx_hash: String },
    /// ZK risk threshold was crossed — ZK proofs are now required.
    ZkRiskActivated { active_workers: u64, min_workers: usize },
    /// Training recovered after cheater removal.
    RecoveryCompleted { honest_workers: usize, resumed_from_step: usize, post_recovery_steps: usize },
}

/// A callback for receiving progress events.
pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

/// Maximum number of retries for chain operations.
const CHAIN_RETRY_MAX: u32 = 3;
/// Delay between retries for chain operations.
const CHAIN_RETRY_DELAY: Duration = Duration::from_secs(2);

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
    /// Layer dimensions, e.g. [784, 128, 10] for MNIST.
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
    /// Mini-batch size for training. Default 1 (standard SGD).
    /// Higher values (e.g. 32) improve CPU utilization by processing
    /// multiple samples per communication round.
    pub batch_size: usize,
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

    // -- Optional ZK Proof Configuration --
    /// Configuration for optional ZK proof generation at checkpoints.
    /// When enabled, generates `StateTransitionCircuit` proofs at checkpoint
    /// boundaries for external verifiability. Default: disabled.
    pub zk_proof: ZkProofConfig,

    /// Controls when ZK proofs are activated for on-chain checkpoint submission.
    ///
    /// - `ZkMode::Off` (default): Pure MPC+MAC attestation, no ZK.
    /// - `ZkMode::Always`: ZK proofs required for every checkpoint.
    /// - `ZkMode::Risk { min_workers }`: Auto-activate ZK when active workers
    ///   drop below `min_workers` (e.g. due to cheater slashing).
    ///
    /// When `ZkMode::Risk` is active and the threshold is crossed mid-training,
    /// the orchestrator dynamically creates a `ZkProofLayer` for remaining
    /// checkpoint submissions.
    #[serde(default)]
    pub zk_mode: ZkMode,

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
    /// Use the on-chain global worker pool (V4 `assignPoolWorkers`) instead of
    /// individual per-worker `stakeAndJoin`.  When true, Phase 6 calls
    /// `assign_pool_workers(job_id, count)` from the owner's wallet, and
    /// `worker_private_keys` are only needed for signing attestations (demo mode).
    #[cfg(feature = "chain")]
    pub use_pool_workers: bool,

    /// Whether to execute the stake withdrawal phase after training completion.
    /// Requires local Anvil (uses `evm_increaseTime` to advance past the 7-day cooldown).
    /// Defaults to false. Enable for demo/integration test scenarios.
    #[cfg(feature = "chain")]
    pub enable_withdrawal: bool,

    /// When set, skip Phase 5 (job registration) and use this pre-registered job ID.
    /// The user's wallet already registered and paid for the job on-chain.
    #[cfg(feature = "chain")]
    #[serde(default)]
    pub pre_registered_job_id: Option<u64>,

    /// HelixModelStore (ERC-721) contract address. When set, a model NFT is minted
    /// after training completion. Pass None to skip NFT minting.
    #[cfg(feature = "chain")]
    #[serde(default)]
    pub model_store_address: Option<String>,

    /// Model slug for the NFT (globally unique identifier). Required when model_store_address is set.
    #[cfg(feature = "chain")]
    #[serde(default)]
    pub model_slug: Option<String>,

    /// Model display name for the NFT. Defaults to "HELIX Model" if not set.
    #[cfg(feature = "chain")]
    #[serde(default)]
    pub model_name: Option<String>,

    /// Model description for the NFT.
    #[cfg(feature = "chain")]
    #[serde(default)]
    pub model_description: Option<String>,

    // -- Transport Mode --
    /// When true, workers run the MPC training loop on their own machines
    /// using TcpTransport (distributed/multi-machine mode). The orchestrator
    /// sends `StartDistributedTraining` to each worker over the data channel.
    /// When false (default), the orchestrator runs all MPC parties in-process
    /// using LocalTransport (single-machine, faster).
    #[serde(default)]
    pub distributed: bool,

    // -- Custom Training Data --
    /// Pre-loaded training data (overrides MNIST generation when set).
    /// Format: Vec of (input_vec, target_vec) pairs.
    #[serde(skip)]
    pub custom_training_data: Option<Vec<(Vec<f64>, Vec<f64>)>>,

    // -- Worker Seeds --
    /// Per-worker seeds for deterministic x25519 key derivation.
    /// worker_seeds[i] is the seed for worker i (used as: seeded_rng(seed, party_index)).
    /// If empty, falls back to self.seed + i (legacy behavior).
    #[serde(default)]
    pub worker_seeds: Vec<u64>,

    // -- Cheater Simulation (demo feature) --
    /// When true, one worker will inject corrupt shares mid-training to demonstrate
    /// cheater detection and slashing. The cheater party and corruption step are
    /// chosen automatically (party 2 at step num_steps/2).
    #[serde(default)]
    pub simulate_cheater: bool,

    /// Which party should cheat (0-indexed). Default: last worker (num_workers - 1).
    /// Only used when simulate_cheater is true.
    #[serde(default)]
    pub cheater_party: Option<usize>,

    /// At which training step the cheater corrupts weights. Default: num_steps / 2.
    /// Only used when simulate_cheater is true.
    #[serde(default)]
    pub cheater_step: Option<u64>,
}

impl Default for FullOrchestrationConfig {
    fn default() -> Self {
        Self {
            architecture: vec![784, 128, 10],
            num_steps: 100,
            learning_rate: 0.01,
            checkpoint_frequency: 10,
            mac_check_interval: 10,
            beaver_batch_size: 2048,
            batch_size: 4,
            seed: 42,
            worker_endpoints: vec![
                "127.0.0.1:9001".to_string(),
                "127.0.0.1:9002".to_string(),
                "127.0.0.1:9003".to_string(),
            ],
            initial_weights_path: None,
            zk_proof: ZkProofConfig::default(),
            zk_mode: ZkMode::Off,
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
            payment_amount_eth: 0.01,
            #[cfg(feature = "chain")]
            stake_amount_eth: 0.001,
            #[cfg(feature = "chain")]
            coordinator_address: None,
            #[cfg(feature = "chain")]
            use_pool_workers: false,
            #[cfg(feature = "chain")]
            enable_withdrawal: false,
            #[cfg(feature = "chain")]
            pre_registered_job_id: None,
            #[cfg(feature = "chain")]
            model_store_address: None,
            #[cfg(feature = "chain")]
            model_slug: None,
            #[cfg(feature = "chain")]
            model_name: None,
            #[cfg(feature = "chain")]
            model_description: None,
            distributed: false,
            custom_training_data: None,
            worker_seeds: Vec::new(),
            simulate_cheater: false,
            cheater_party: None,
            cheater_step: None,
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
    /// Number of ZK proofs generated (0 if ZK disabled).
    pub zk_proofs_generated: usize,
    /// Number of ZK proofs submitted on-chain (0 if ZK disabled or no chain).
    pub zk_proofs_on_chain: usize,
    /// Model NFT token ID (None if NFT minting was not configured or failed).
    pub model_nft_token_id: Option<u64>,
    /// Model store contract address (empty if not configured).
    pub model_store_address: String,
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
    /// Optional progress callback for user-facing display.
    progress: Option<ProgressCallback>,
    /// Optional ZK proof layer.
    /// Initialized at startup when `zk_proof.enabled == true` or `zk_mode == ZkMode::Always`.
    /// For `ZkMode::Risk`, created dynamically in Phase 10.5 if the risk threshold is crossed.
    zk_layer: Option<ZkProofLayer>,
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
        // Create the ZK layer if explicitly enabled or if zk_mode is Always.
        // For ZkMode::Risk, the layer is created dynamically in Phase 10.5
        // only if the risk threshold is actually crossed.
        let should_init_zk = config.zk_proof.enabled
            || matches!(config.zk_mode, ZkMode::Always);

        let zk_layer = if should_init_zk && config.architecture.len() == 3 {
            let d_in = config.architecture[0];
            let d_hid = config.architecture[1];
            let d_out = config.architecture[2];

            let mut zk_config = config.zk_proof.clone();
            if matches!(config.zk_mode, ZkMode::Always) {
                zk_config.enabled = true;
                zk_config.checkpoint_frequency = 1; // Prove every checkpoint in Always mode
            }

            Some(ZkProofLayer::new(zk_config, d_in, d_hid, d_out))
        } else {
            None
        };

        Self {
            config,
            progress: None,
            zk_layer,
            #[cfg(feature = "chain")]
            anvil_process: None,
            #[cfg(feature = "chain")]
            rpc_url: None,
        }
    }

    /// Sets a progress callback for user-facing status updates.
    ///
    /// The callback is invoked at key moments during orchestration so the CLI
    /// can render formatted output without coupling to the display mechanism.
    pub fn set_progress_callback(&mut self, cb: ProgressCallback) {
        self.progress = Some(cb);
    }

    /// Emits a progress event to the registered callback (if any).
    fn emit(&self, event: ProgressEvent) {
        if let Some(ref cb) = self.progress {
            cb(event);
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
        self.emit(ProgressEvent::PhaseStarted {
            phase: 1, total: 13,
            description: format!(
                "Loading MNIST training data ({} samples)",
                self.config.train_size,
            ),
        });
        info!("Phase 1: Loading training data");
        let phase1_start = Instant::now();

        let dataset = self
            .load_dataset(d_in, d_out)
            .context("Phase 1: Failed to load training data")?;

        let training_pairs = MnistDataset::as_training_pairs(&dataset.train);
        let test_samples = dataset.test.clone();

        let phase1_elapsed = phase1_start.elapsed().as_millis();
        info!(
            train_samples = training_pairs.len(),
            test_samples = test_samples.len(),
            elapsed_ms = phase1_elapsed,
            "Phase 1 complete: training data loaded"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 1, elapsed_ms: phase1_elapsed });

        // ================================================================
        // Phase 2: Generate or load initial weights
        // ================================================================
        let weight_source = if self.config.initial_weights_path.is_some() { "file" } else { "xavier" };
        self.emit(ProgressEvent::PhaseStarted {
            phase: 2, total: 13,
            description: format!(
                "Initializing model weights ({}x{}x{}, {})",
                d_in, d_hid, d_out, weight_source,
            ),
        });
        info!("Phase 2: Initializing model weights");
        let phase2_start = Instant::now();

        let initial_weights = self
            .load_or_generate_weights(d_in, d_hid, d_out)
            .context("Phase 2: Failed to initialize weights")?;

        let total_params = d_hid * d_in + d_hid + d_out * d_hid + d_out;
        let phase2_elapsed = phase2_start.elapsed().as_millis();
        info!(
            total_params = total_params,
            source = if self.config.initial_weights_path.is_some() { "file" } else { "xavier" },
            elapsed_ms = phase2_elapsed,
            "Phase 2 complete: weights initialized"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 2, elapsed_ms: phase2_elapsed });

        // Set ZK baseline weights from initial model state (before MPC training).
        if let Some(ref mut zk_layer) = self.zk_layer {
            zk_layer.set_baseline_weights(
                &initial_weights.w1,
                &initial_weights.b1,
                &initial_weights.w2,
                &initial_weights.b2,
            ).context("Failed to set ZK baseline weights")?;
        }

        // ================================================================
        // Phases 3-7: On-chain setup (feature-gated)
        // Skip entirely when payment_amount_eth == 0 (local dev / no on-chain registration)
        // ================================================================
        #[cfg(feature = "chain")]
        let skip_chain = self.config.payment_amount_eth == 0.0
            && self.config.pre_registered_job_id.is_none();

        #[cfg(feature = "chain")]
        let (job_id, coordinator_address, chain_client, worker_wallets, chain_gas) = if skip_chain {
            info!("Skipping on-chain phases (payment=0, no pre-registered job). Using local-only mode.");
            for phase in 3..=7 {
                self.emit(ProgressEvent::PhaseStarted {
                    phase, total: 13,
                    description: format!("Phase {} skipped (local mode)", phase),
                });
                self.emit(ProgressEvent::PhaseCompleted { phase, elapsed_ms: 0 });
            }
            // No chain client needed — job_id=0 gates all downstream chain usage
            // (settlement and withdrawal phases both check job_id==0 and skip).
            let coord_addr = self.config.coordinator_address.clone().unwrap_or_default();
            (0u64, coord_addr, None, Vec::new(), 0u64)
        } else {
            let (jid, addr, client, wallets, gas) =
                self.run_chain_setup_phases(d_in, d_hid, d_out, num_workers).await?;
            (jid, addr, Some(client), wallets, gas)
        };

        #[cfg(not(feature = "chain"))]
        let (job_id, coordinator_address, chain_gas): (u64, String, u64) =
            (0, String::new(), 0);

        // ================================================================
        // Phase 8: Run MPC training
        // ================================================================
        let total_params = d_hid * d_in + d_hid + d_out * d_hid + d_out;
        self.emit(ProgressEvent::PhaseStarted {
            phase: 8, total: 13,
            description: format!(
                "Running MPC training — {} workers, {} params",
                num_workers, total_params,
            ),
        });
        info!("Phase 8: Running MPC training");
        let phase8_start = Instant::now();

        let mpc_result = if self.config.distributed {
            // Distributed mode: workers run the MPC training loop themselves.
            // The orchestrator distributes shares + training config to each worker
            // over the data channel. Workers create TcpTransport meshes, run
            // training, and send results back.
            info!(
                distributed = true,
                num_workers = num_workers,
                "Phase 8: Distributed MPC training (workers execute on their own machines)"
            );
            self.run_distributed_training(
                d_in, d_hid, d_out, num_workers,
                initial_weights, training_pairs,
            ).await
            .context("Phase 8: Distributed MPC training failed")?
        } else {
            // Local mode: all MPC parties run in this process using LocalTransport.
            // Clone the progress callback so live per-step events fire during training.
            let progress_for_step = self.progress.clone();
            let progress_for_sub_step = self.progress.clone();
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
                use_tcp_transport: false,
                worker_endpoints: None,
                batch_size: self.config.batch_size,
                on_step: progress_for_step.map(|cb| -> std::sync::Arc<dyn Fn(usize, usize, f64, f64, bool) + Send + Sync> {
                    std::sync::Arc::new(move |step, total, loss, accuracy, mac_ok| {
                        cb(ProgressEvent::TrainingStep { step, total, loss, accuracy, mac_ok });
                    })
                }),
                on_sub_step: progress_for_sub_step.map(|cb| -> std::sync::Arc<dyn Fn(usize, usize, &str) + Send + Sync> {
                    std::sync::Arc::new(move |step, total, operation| {
                        cb(ProgressEvent::SubStep {
                            step,
                            total,
                            operation: operation.to_string(),
                        });
                    })
                }),
            };

            if self.config.simulate_cheater {
                let cheater_party = self.config.cheater_party.unwrap_or(num_workers - 1);
                let corrupt_at_step = self.config.cheater_step.unwrap_or((self.config.num_steps / 2) as u64);
                info!(
                    cheater_party = cheater_party,
                    corrupt_at_step = corrupt_at_step,
                    "Simulating cheater injection for demo"
                );
                helix_mpc::e2e_integration::run_mpc_training_with_cheater(
                    mpc_config,
                    cheater_party,
                    corrupt_at_step,
                )
                .await
                .context("Phase 8: MPC training with cheater simulation failed")?
            } else {
                helix_mpc::e2e_integration::run_mpc_training(mpc_config)
                    .await
                    .context("Phase 8: MPC training failed")?
            }
        };

        // Note: per-step progress for local mode is now emitted live via the
        // on_step callback wired into the MPC config above.

        if let Some(ref cheater) = mpc_result.cheater_detected {
            self.emit(ProgressEvent::CheaterDetected {
                party_index: cheater.party_index,
                step: cheater.detected_at_step,
            });
        }

        if mpc_result.recovery_completed {
            let resumed_from = mpc_result.steps_completed.saturating_sub(mpc_result.post_recovery_steps);
            self.emit(ProgressEvent::RecoveryCompleted {
                honest_workers: num_workers - 1,
                resumed_from_step: resumed_from,
                post_recovery_steps: mpc_result.post_recovery_steps,
            });
        }

        let phase8_elapsed = phase8_start.elapsed().as_millis();
        info!(
            steps_completed = mpc_result.steps_completed,
            final_loss = mpc_result.final_loss,
            checkpoints = mpc_result.checkpoints.len(),
            mac_checks_passed = mpc_result.mac_checks_passed,
            cheater_detected = mpc_result.cheater_detected.is_some(),
            encrypted_distribution = mpc_result.encrypted_distribution,
            elapsed_ms = phase8_elapsed,
            "Phase 8 complete: MPC training finished"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 8, elapsed_ms: phase8_elapsed });

        // ================================================================
        // Phases 9-11: On-chain settlement (feature-gated)
        // ================================================================
        #[cfg(feature = "chain")]
        let (checkpoints_on_chain, cheater_info, settlement_gas, zk_proofs_on_chain) = if job_id == 0 {
            // Local-only mode (skip_chain): no on-chain settlement.
            info!("Skipping on-chain settlement phases (local mode, job_id=0)");
            for phase in 9..=11 {
                self.emit(ProgressEvent::PhaseStarted {
                    phase, total: 13,
                    description: format!("Phase {} skipped (local mode)", phase),
                });
                self.emit(ProgressEvent::PhaseCompleted { phase, elapsed_ms: 0 });
            }
            let ci = mpc_result.cheater_detected.as_ref().map(|c| CheaterInfo {
                party_index: c.party_index,
                detected_at_step: c.detected_at_step,
                slashed: false,
                slash_tx_hash: None,
            });
            (0, ci, 0u64, 0usize)
        } else {
            self.run_chain_settlement_phases(
                job_id,
                &mpc_result,
                chain_client.as_ref().expect("chain_client must exist when job_id != 0"),
                &worker_wallets,
            )
            .await?
        };

        #[cfg(not(feature = "chain"))]
        let (checkpoints_on_chain, cheater_info, settlement_gas, zk_proofs_on_chain): (usize, Option<CheaterInfo>, u64, usize) = {
            let ci = mpc_result.cheater_detected.as_ref().map(|c| CheaterInfo {
                party_index: c.party_index,
                detected_at_step: c.detected_at_step,
                slashed: false,
                slash_tx_hash: None,
            });
            (0, ci, 0, 0)
        };

        // ================================================================
        // Phase 11.5: Stake withdrawal (feature-gated, opt-in)
        // ================================================================
        #[cfg(feature = "chain")]
        let withdrawal_gas = if self.config.enable_withdrawal {
            self.run_withdrawal_phase(
                job_id,
                &coordinator_address,
                &worker_wallets,
                &cheater_info,
            )
            .await
            .unwrap_or_else(|e| {
                warn!("Phase 11.5: Stake withdrawal failed: {}. Continuing.", e);
                0
            })
        } else {
            info!("Phase 11.5: Stake withdrawal skipped (enable_withdrawal=false)");
            0
        };
        #[cfg(not(feature = "chain"))]
        let withdrawal_gas: u64 = 0;

        // ================================================================
        // Phase 12: Evaluate accuracy on test set
        // ================================================================
        self.emit(ProgressEvent::PhaseStarted {
            phase: 12, total: 13,
            description: format!("Evaluating accuracy on {} test samples", test_samples.len()),
        });
        info!("Phase 12: Evaluating test accuracy");
        let phase12_start = Instant::now();

        let test_accuracy = evaluate_accuracy(&mpc_result.final_weights, &test_samples, d_in, d_hid, d_out);

        let phase12_elapsed = phase12_start.elapsed().as_millis();
        info!(
            test_accuracy = format!("{:.2}%", test_accuracy * 100.0),
            test_samples = test_samples.len(),
            elapsed_ms = phase12_elapsed,
            "Phase 12 complete: accuracy evaluated"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 12, elapsed_ms: phase12_elapsed });

        // ================================================================
        // Phase 12.5: Mint Model NFT (feature-gated, opt-in)
        // ================================================================
        #[cfg(feature = "chain")]
        let (model_nft_token_id, model_store_addr_str) = if let Some(ref store_addr) = self.config.model_store_address {
            if job_id != 0 {
                info!("Phase 12.5: Minting model NFT");
                let slug = self.config.model_slug.clone().unwrap_or_else(|| {
                    format!("helix-model-{:x}", job_id)
                });
                let name = self.config.model_name.clone().unwrap_or_else(|| {
                    format!("HELIX Model (Job {})", job_id)
                });
                let description = self.config.model_description.clone().unwrap_or_else(|| {
                    format!(
                        "Trained with HELIX MPC protocol. Architecture: {}x{}x{}, {} steps, {:.1}% accuracy",
                        d_in, d_hid, d_out, mpc_result.steps_completed, test_accuracy * 100.0
                    )
                });

                let rpc = self.rpc_url.as_ref().ok_or_else(|| anyhow!("No RPC URL for NFT minting"))?;
                match ModelStoreClient::new(rpc, &self.config.private_key, store_addr, None).await {
                    Ok(store_client) => {
                        let architecture = format!("{}x{}x{}", d_in, d_hid, d_out);
                match store_client.create_model(&slug, &name, &description, &architecture).await {
                            Ok(create_result) => {
                                let token_id = create_result.token_id;
                                info!(
                                    token_id = token_id,
                                    slug = %slug,
                                    tx = %format!("{:?}", create_result.receipt.transaction_hash),
                                    "Model NFT minted"
                                );

                                // Add version with accuracy
                                let accuracy_scaled = (test_accuracy * 10000.0) as u64;
                                let session_id = format!("job-{}", job_id);
                                match store_client.add_version(
                                    token_id, "1.0.0", "", accuracy_scaled, &session_id, false,
                                ).await {
                                    Ok(receipt) => {
                                        info!(
                                            tx = %format!("{:?}", receipt.transaction_hash),
                                            accuracy = format!("{:.2}%", test_accuracy * 100.0),
                                            "Version added to model NFT"
                                        );
                                    }
                                    Err(e) => warn!("Failed to add version to NFT: {}", e),
                                }

                                (Some(token_id), store_addr.clone())
                            }
                            Err(e) => {
                                warn!("Phase 12.5: Failed to mint model NFT: {}. Continuing.", e);
                                (None, store_addr.clone())
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Phase 12.5: Failed to connect to ModelStore: {}. Continuing.", e);
                        (None, store_addr.clone())
                    }
                }
            } else {
                info!("Phase 12.5: Skipping NFT minting (local mode, job_id=0)");
                (None, store_addr.clone())
            }
        } else {
            (None, String::new())
        };

        #[cfg(not(feature = "chain"))]
        let (model_nft_token_id, model_store_addr_str): (Option<u64>, String) = (None, String::new());

        // ================================================================
        // Phase 13: Print comprehensive summary
        // ================================================================
        let total_elapsed = overall_start.elapsed().as_secs_f64();

        let total_gas = chain_gas + settlement_gas + withdrawal_gas;

        let zk_proofs_generated = self
            .zk_layer
            .as_ref()
            .map(|zk| zk.proofs_generated())
            .unwrap_or(0);

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
            zk_proofs_generated,
            zk_proofs_on_chain,
            model_nft_token_id,
            model_store_address: model_store_addr_str,
        };

        self.emit(ProgressEvent::PhaseStarted {
            phase: 13, total: 13,
            description: "Training complete".to_string(),
        });
        self.print_summary(&result);
        self.emit(ProgressEvent::TrainingComplete {
            accuracy: result.test_accuracy,
            steps: result.steps_completed,
            time_secs: result.training_time_secs,
            checkpoints: result.checkpoints_on_chain,
        });
        self.emit(ProgressEvent::PhaseCompleted { phase: 13, elapsed_ms: 0 });

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
    // Distributed Training
    // ========================================================================

    /// Runs MPC training in distributed mode where workers actively execute
    /// the training loop on their own machines.
    ///
    /// Flow:
    /// 1. Prepare encrypted shares for each worker
    /// 2. Connect to each worker's data channel and distribute shares
    /// 3. Send `StartDistributedTraining` to each worker with peer addresses,
    ///    trainer config, and training data
    /// 4. Collect `DistributedTrainingResult` from each worker
    /// 5. Reconstruct weights from encrypted final shares
    async fn run_distributed_training(
        &self,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        num_workers: usize,
        initial_weights: InitialWeights,
        training_data: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> Result<MPCIntegrationResult> {
        use helix_mpc::e2e_integration::{
            prepare_encrypted_shares, WeightLayout, FinalWeights, CheckpointRecord,
        };
        use helix_mpc::mpc_trainer::MPCTrainerConfig;
        use helix_mpc::mac_verification::MACVerificationConfig;
        use helix_mpc::network_distribution::{
            distribute_shares, recv_message, send_message, ProtocolMessage,
        };
        use helix_mpc::share_distribution::{
            generate_x25519_keypair, WeightReconstructor, X25519PublicKey,
        };
        use helix_mpc::types::PartyId;
        use std::net::SocketAddr;

        let start = std::time::Instant::now();

        // Build the trainer config that workers will use.
        let trainer_config = MPCTrainerConfig {
            d_in,
            d_hid,
            d_out,
            learning_rate: self.config.learning_rate,
            num_parties: num_workers,
            reshare_interval: 0,
            beaver_batch_size: self.config.beaver_batch_size,
            generate_proofs: false,
            base_error: 1e-6,
            checkpoint_interval: self.config.checkpoint_frequency as u64,
            mac_config: if self.config.mac_check_interval > 0 {
                Some(MACVerificationConfig {
                    check_interval: self.config.mac_check_interval,
                    enable_cheater_identification: true,
                    mac_seed: self.config.seed.wrapping_mul(0xCAFE_BABE),
                })
            } else {
                None
            },
            batch_size: self.config.batch_size.max(1),
        };

        let weight_layout = WeightLayout::new(d_in, d_hid, d_out);

        // Generate owner x25519 keypair for encrypting final shares.
        let mut key_rng = rand::rngs::StdRng::seed_from_u64(
            self.config.seed.wrapping_mul(0xDEAD_BEEF),
        );
        let (owner_secret, owner_public) = generate_x25519_keypair(&mut key_rng);

        // Parse worker endpoints and compute MPC ports (data_port + 2).
        let worker_addrs: Vec<SocketAddr> = self.config.worker_endpoints.iter()
            .map(|ep| ep.parse::<SocketAddr>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid worker endpoint address")?;

        let mpc_addrs: Vec<SocketAddr> = worker_addrs.iter()
            .map(|addr| SocketAddr::new(addr.ip(), addr.port() + 2))
            .collect();

        let parties: Vec<PartyId> = (0..num_workers).map(PartyId::from_index).collect();

        // Build peer address list: (party_id_str, mpc_addr_str) for all parties.
        let all_peer_addrs: Vec<(String, String)> = parties.iter()
            .zip(mpc_addrs.iter())
            .map(|(pid, addr)| (pid.to_string(), addr.to_string()))
            .collect();

        // Compute worker x25519 public keys using the same derivation as workers.
        // Each worker generates its key with seeded_rng(seed, party_index).
        // The seed comes from the worker's registration (worker_seeds), NOT the
        // training session seed. This ensures key agreement across sessions.
        let mut worker_publics = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let worker_seed = self.config.worker_seeds.get(i)
                .copied()
                .unwrap_or(self.config.seed + i as u64);
            let pk_bytes = crate::worker_entry::worker_public_key_from_seed(
                worker_seed, i,
            );
            worker_publics.push(X25519PublicKey::from(pk_bytes));
        }

        // Distribute encrypted shares to workers via TCP.
        info!(
            num_workers = num_workers,
            "Distributing encrypted shares to workers"
        );

        let worker_info: Vec<(PartyId, SocketAddr, X25519PublicKey)> = parties.iter()
            .zip(worker_addrs.iter())
            .zip(worker_publics.iter())
            .map(|((pid, addr), pk)| (pid.clone(), *addr, *pk))
            .collect();

        // Convert initial weights to flat f64 for distribution.
        let flat_weights: Vec<f64> = initial_weights.w1.iter()
            .chain(initial_weights.b1.iter())
            .chain(initial_weights.w2.iter())
            .chain(initial_weights.b2.iter())
            .copied()
            .collect();

        let shape = vec![flat_weights.len()];

        let mut dist_result = distribute_shares(
            &flat_weights,
            &shape,
            &worker_info,
            Some(self.config.seed),
        ).await.map_err(|e| anyhow!("Share distribution failed: {}", e))?;

        if !dist_result.verified {
            return Err(anyhow!("Commitment verification failed after share distribution"));
        }
        info!("Encrypted shares distributed and commitment verified");

        // Connect to each worker's control channel (data_port + 1) BEFORE
        // sending StartDistributedTraining. Workers bind their control port
        // after receiving shares, so we retry briefly if not yet ready. We must
        // connect here first because workers block on control_accept before
        // reading the next data channel message. If we sent
        // StartDistributedTraining first, the large message (~6 MB training
        // data) would fill TCP buffers while workers aren't reading, causing a
        // deadlock.
        let mut control_streams = Vec::with_capacity(num_workers);
        for (i, addr) in worker_addrs.iter().enumerate() {
            let control_addr = SocketAddr::new(addr.ip(), addr.port() + 1);
            let mut stream_opt = None;
            for attempt in 0..10 {
                match tokio::time::timeout(
                    Duration::from_secs(2),
                    tokio::net::TcpStream::connect(control_addr),
                ).await {
                    Ok(Ok(s)) => {
                        stream_opt = Some(s);
                        break;
                    }
                    _ => {
                        if attempt < 9 {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                        }
                    }
                }
            }
            let stream = stream_opt.ok_or_else(|| {
                anyhow!("Failed to connect to worker {} control channel at {} after 10 attempts", i, control_addr)
            })?;
            info!(worker = i, addr = %control_addr, "Connected to worker control channel");
            control_streams.push(stream);
        }
        info!("Control channels connected to all {} workers", num_workers);

        // Now send StartDistributedTraining to each worker over the open data
        // channel. Workers have completed control_accept and are now waiting on
        // recv_message, so the data channel writes will not block.
        for (i, (_party_id, stream)) in dist_result.worker_streams.iter_mut().enumerate() {
            let msg = ProtocolMessage::StartDistributedTraining {
                trainer_config: trainer_config.clone(),
                peer_addrs: all_peer_addrs.clone(),
                training_data: training_data.clone(),
                weight_layout: weight_layout.clone(),
                owner_public_key: *owner_public.as_bytes(),
                num_steps: self.config.num_steps,
                checkpoint_interval: self.config.checkpoint_frequency,
                seed: self.config.seed,
                mpc_bind_addr: mpc_addrs[i].to_string(),
            };
            send_message(stream, &msg).await
                .map_err(|e| anyhow!("Failed to send training command to worker {}: {}", i, e))?;
        }
        info!("StartDistributedTraining sent to all {} workers", num_workers);

        // Collect results from all workers.
        // Worker 0 sends StepProgress messages during training (for live dashboard
        // updates) followed by DistributedTrainingResult. Other workers only send
        // the final result.
        let mut all_encrypted_shares = Vec::with_capacity(num_workers);
        let mut all_losses = Vec::new();
        let mut total_steps = 0;
        let mut total_mac_checks = 0;
        let mut cheater_detected: Option<CheaterRecord> = None;
        let mut all_checkpoints = Vec::new();

        for (i, (_party_id, stream)) in dist_result.worker_streams.iter_mut().enumerate() {
            // Worker 0: read StepProgress messages until we get the final result.
            // This streams live loss/accuracy to the dashboard via WebSocket.
            let result_msg = if i == 0 {
                loop {
                    let msg = recv_message(stream).await
                        .map_err(|e| anyhow!("Failed to receive message from worker 0: {}", e))?;
                    match msg {
                        ProtocolMessage::StepProgress { step, total, loss, accuracy, mac_ok } => {
                            self.emit(ProgressEvent::TrainingStep {
                                step, total, loss, accuracy, mac_ok,
                            });
                        }
                        other => break other,
                    }
                }
            } else {
                recv_message(stream).await
                    .map_err(|e| anyhow!("Failed to receive result from worker {}: {}", i, e))?
            };

            match result_msg {
                ProtocolMessage::DistributedTrainingResult {
                    steps_completed,
                    losses,
                    mac_checks_passed,
                    cheater_detected: has_cheater,
                    cheater_party,
                    cheater_detected_at_step,
                    mac_failure_report,
                    encrypted_final_share,
                    checkpoints,
                } => {
                    info!(
                        worker = i,
                        steps = steps_completed,
                        final_loss = losses.last().copied().unwrap_or(0.0),
                        mac_checks = mac_checks_passed,
                        cheater = has_cheater,
                        "Worker {} training result received",
                        i
                    );

                    if i == 0 {
                        all_losses = losses;
                        total_steps = steps_completed;
                        total_mac_checks = mac_checks_passed;
                        all_checkpoints = checkpoints.iter().map(|cp| {
                            CheckpointRecord {
                                step: cp.step,
                                commitment_bytes32: cp.commitment_bytes32,
                                loss: cp.loss,
                            }
                        }).collect();
                    }

                    if has_cheater {
                        // Use the actual failure report from the worker if available,
                        // falling back to a skeleton if the worker didn't send one.
                        let failure_report = mac_failure_report.unwrap_or_else(|| {
                            helix_mpc::mac_verification::MACFailureReport {
                                session_id: "distributed".to_string(),
                                step_number: cheater_detected_at_step,
                                identified_cheater: cheater_party,
                                sigma_values: Vec::new(),
                                commitments: Vec::new(),
                                evidence: helix_mpc::mac_verification::CheaterEvidence {
                                    pairwise_results: Vec::new(),
                                    round1_sigmas: Vec::new(),
                                    round2_sigmas: Vec::new(),
                                },
                            }
                        });
                        cheater_detected = Some(CheaterRecord {
                            party_index: cheater_party.unwrap_or(usize::MAX),
                            detected_at_step: cheater_detected_at_step,
                            failure_report,
                        });
                    }

                    if let Some(enc_share) = encrypted_final_share {
                        all_encrypted_shares.push(enc_share);
                    }
                }
                other => {
                    return Err(anyhow!(
                        "Unexpected message from worker {}: expected DistributedTrainingResult, got {:?}",
                        i, std::mem::discriminant(&other),
                    ));
                }
            }
        }

        // Reconstruct final weights from encrypted shares.
        let final_weights = if !all_encrypted_shares.is_empty() {
            let reconstructor = WeightReconstructor::new(owner_secret);
            let flat = reconstructor.reconstruct(&all_encrypted_shares, None, None)
                .map_err(|e| anyhow!("Weight reconstruction failed: {}", e))?;

            let mut offset = 0;
            let w1 = flat[offset..offset + weight_layout.w1_len].to_vec();
            offset += weight_layout.w1_len;
            let b1 = flat[offset..offset + weight_layout.b1_len].to_vec();
            offset += weight_layout.b1_len;
            let w2 = flat[offset..offset + weight_layout.w2_len].to_vec();
            offset += weight_layout.w2_len;
            let b2 = flat[offset..offset + weight_layout.b2_len].to_vec();

            info!("Final weights reconstructed from {} encrypted shares", all_encrypted_shares.len());
            FinalWeights { w1, b1, w2, b2 }
        } else {
            return Err(anyhow!("No encrypted final shares received from workers"));
        };

        let training_time_ms = start.elapsed().as_millis();
        let final_loss = all_losses.last().copied().unwrap_or(0.0);

        Ok(MPCIntegrationResult {
            steps_completed: total_steps,
            final_loss,
            losses: all_losses,
            checkpoints: all_checkpoints,
            mac_checks_passed: total_mac_checks,
            cheater_detected,
            training_time_ms,
            final_weights,
            initial_commitment: Some(dist_result.distribution.initial_commitment),
            encrypted_distribution: true,
            recovery_completed: false,
            post_recovery_losses: Vec::new(),
            post_recovery_steps: 0,
        })
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
            // When using the on-chain pool, worker_private_keys are still needed for
            // demo-mode signing but we don't require a 1:1 match since the pool
            // dynamically assigns workers.
            if !self.config.use_pool_workers
                && self.config.worker_private_keys.len() != self.config.worker_endpoints.len()
            {
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
        // If custom training data was provided, convert to MnistDataset format
        if let Some(ref custom_data) = self.config.custom_training_data {
            info!(
                samples = custom_data.len(),
                "Using custom uploaded training data"
            );
            let mut samples: Vec<MnistSample> = custom_data
                .iter()
                .map(|(input, target)| MnistSample {
                    pixels: input.clone(),
                    label: target.clone(),
                    digit: target.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0),
                })
                .collect();

            // Split into train/test (use configured test_size, capped at 20% of data)
            let test_count = self.config.test_size.min(samples.len() / 5).max(1);
            let train_count = samples.len() - test_count;
            let test = samples.split_off(train_count);

            return Ok(MnistDataset {
                train: samples,
                test,
            });
        }

        if self.config.use_real_mnist {
            let cache_dir = self.config.mnist_cache_dir.as_ref().map(std::path::Path::new);
            info!(
                train_size = self.config.train_size,
                test_size = self.config.test_size,
                seed = self.config.seed,
                cache_dir = ?cache_dir,
                "Loading real MNIST dataset"
            );
            let dataset = MnistDataset::load_real_shuffled(
                self.config.train_size,
                self.config.test_size,
                self.config.seed,
                cache_dir,
            ).map_err(|e| anyhow::anyhow!("Failed to load real MNIST: {}", e))?;
            info!(
                train = dataset.train.len(),
                test = dataset.test.len(),
                "Real MNIST loaded successfully"
            );
            Ok(dataset)
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
        self.emit(ProgressEvent::PhaseStarted {
            phase: 3, total: 13,
            description: "Bootstrapping chain connection".to_string(),
        });
        let phase3_start = Instant::now();
        let rpc_url = if let Some(ref url) = self.config.eth_rpc_url {
            // Verify RPC is reachable before proceeding.
            info!(rpc_url = %url, "Phase 3: Verifying provided RPC URL");
            let provider_check = Provider::<Http>::try_from(url.as_str())
                .map_err(|e| anyhow!("Phase 3: Invalid RPC URL '{}': {}", url, e))?;
            match provider_check.get_chainid().await {
                Ok(chain_id) => {
                    info!(chain_id = chain_id.as_u64(), "RPC connection verified");
                }
                Err(e) => {
                    return Err(anyhow!(
                        "Phase 3: Cannot connect to RPC at '{}'. \
                         Check the URL and ensure the node is running. Error: {}",
                        url, e
                    ));
                }
            }
            url.clone()
        } else {
            info!("Phase 3: Starting local Anvil instance");
            let url = self.start_anvil().context(
                "Phase 3: Failed to start Anvil. Is Foundry installed? \
                 Install with: curl -L https://foundry.paradigm.xyz | bash && foundryup"
            )?;
            info!(
                rpc_url = %url,
                elapsed_ms = phase3_start.elapsed().as_millis(),
                "Phase 3 complete: Anvil running"
            );
            url
        };
        self.rpc_url = Some(rpc_url.clone());
        self.emit(ProgressEvent::PhaseCompleted { phase: 3, elapsed_ms: phase3_start.elapsed().as_millis() });

        // -- Phase 4: Deploy V4 contract if needed --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 4, total: 13,
            description: "Deploying contracts".to_string(),
        });
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

            // Deploy a verifier when ZK proofs may be needed
            // (enabled explicitly, Always mode, or Risk mode which might activate later).
            // Otherwise use Address::zero() to disable on-chain ZK verification.
            //
            // Deploy the real Halo2Verifier (wraps Halo2VerifierCore + Halo2VerifyingKey
            // generated from the StateTransitionCircuit with 6 public inputs).
            let needs_verifier = self.config.zk_proof.enabled
                || matches!(self.config.zk_mode, ZkMode::Always | ZkMode::Risk { .. });
            let verifier = if needs_verifier {
                info!("Phase 4: Deploying MockVerifier for on-chain ZK proof acceptance");
                info!("  (ZK proofs are generated & verified off-chain; MockVerifier accepts on-chain)");

                // Create a temporary client for deploying the verifier.
                // We use MockVerifier because the Halo2Verifier's embedded verifying key
                // must exactly match the circuit parameters. The proof is already verified
                // locally (verified=true), so MockVerifier lets the on-chain flow complete
                // while demonstrating real ZK proof generation.
                let temp_provider = Provider::<Http>::try_from(rpc_url.as_str())
                    .map_err(|e| anyhow!("Invalid RPC URL for verifier deploy: {}", e))?;
                let temp_wallet = LocalWallet::from_str(owner_pk)
                    .map_err(|e| anyhow!("Invalid owner key for verifier deploy: {}", e))?
                    .with_chain_id(
                        temp_provider.get_chainid().await
                            .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                            .as_u64()
                    );
                let temp_client = Arc::new(SignerMiddleware::new(temp_provider, temp_wallet));

                let verifier_contract = crate::rpc::chain_v4::MockVerifierContract::deploy(
                    temp_client, ()
                )
                .map_err(|e| anyhow!("MockVerifier deploy prepare: {}", e))?
                .send()
                .await
                .map_err(|e| anyhow!("MockVerifier deploy send: {}", e))?;

                let verifier_addr = verifier_contract.address();
                info!(
                    verifier_address = %format!("{:?}", verifier_addr),
                    "MockVerifier deployed (ZK proofs verified off-chain, accepted on-chain)"
                );
                verifier_addr
            } else {
                Address::zero() // No ZK verifier for MPC-primary mode
            };

            let (client, deploy_result) = retry_chain_op(
                "contract deployment",
                || async {
                    ChainClientV4::deploy(
                        &rpc_url,
                        &self.config.private_key,
                        treasury,
                        verifier,
                        None,
                    ).await
                },
                &self.progress,
                4,
            ).await
            .context("Phase 4: V4 contract deployment failed after retries")?;

            info!(
                coordinator = %deploy_result.coordinator,
                treasury = %deploy_result.treasury,
                elapsed_ms = phase4_start.elapsed().as_millis(),
                "Phase 4 complete: V4 coordinator deployed"
            );

            (client, deploy_result.coordinator)
        };
        self.emit(ProgressEvent::PhaseCompleted { phase: 4, elapsed_ms: phase4_start.elapsed().as_millis() });

        // -- Phase 5: Register training job --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 5, total: 13,
            description: "Registering training job on-chain".to_string(),
        });
        let phase5_start = Instant::now();

        let arch_hash = compute_architecture_hash(d_in, d_hid, d_out);
        let payment_wei = ethers::utils::parse_ether(self.config.payment_amount_eth)
            .context("Phase 5: Invalid payment amount")?;
        let num_checkpoints = self.config.num_steps / self.config.checkpoint_frequency;

        let job_id = if let Some(pre_id) = self.config.pre_registered_job_id {
            // Job already registered by user's wallet — skip on-chain registration.
            info!(
                job_id = pre_id,
                "Phase 5: Using pre-registered job (user wallet paid)"
            );
            pre_id
        } else {
            info!("Phase 5: Registering training job");

            // Check owner balance before attempting registration.
            {
                let provider = Provider::<Http>::try_from(rpc_url.as_str())
                    .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;
                let owner_pk = self.config.private_key.strip_prefix("0x")
                    .unwrap_or(&self.config.private_key);
                let owner_wallet = LocalWallet::from_str(owner_pk)
                    .map_err(|e| anyhow!("Invalid owner private key: {}", e))?;
                let balance = provider.get_balance(owner_wallet.address(), None).await
                    .context("Phase 5: Failed to check owner balance")?;
                if balance < payment_wei {
                    return Err(anyhow!(
                        "Phase 5: Insufficient funds. Owner balance is {} wei but payment requires {} wei. \
                         Fund the owner address {:?} before starting.",
                        balance, payment_wei, owner_wallet.address()
                    ));
                }
            }

            // Derive ZK parameters from zk_mode for the on-chain registration.
            let zk_enabled = matches!(self.config.zk_mode, ZkMode::Always);
            let risk_zk_enabled = matches!(self.config.zk_mode, ZkMode::Risk { .. });
            let min_workers_for_mpc = match self.config.zk_mode {
                ZkMode::Risk { min_workers } => min_workers as u64,
                _ => 0,
            };
            let zk_checkpoint_freq: u64 = 0;

            info!(
                zk_mode = ?self.config.zk_mode,
                zk_enabled = zk_enabled,
                risk_zk_enabled = risk_zk_enabled,
                min_workers_for_mpc = min_workers_for_mpc,
                zk_checkpoint_freq = zk_checkpoint_freq,
                "Phase 5: Registering job with ZK configuration"
            );

            // Pass the backend's own address as operator so it can call assignPoolWorkers.
            let operator_addr = chain_client.signer_address();

            let (reg_receipt, registered_job_id) = retry_chain_op(
                "job registration",
                || async {
                    chain_client
                        .register_training_job_with_zk(
                            arch_hash,
                            self.config.checkpoint_frequency as u64,
                            self.config.num_steps as u64,
                            payment_wei,
                            zk_enabled,
                            zk_checkpoint_freq,
                            risk_zk_enabled,
                            min_workers_for_mpc,
                            operator_addr,
                        )
                        .await
                },
                &self.progress,
                5,
            ).await
            .context("Phase 5: Job registration failed after retries")?;

            let reg_gas = reg_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += reg_gas;

            info!(
                job_id = registered_job_id,
                gas_used = reg_gas,
                "Phase 5: Job registered on-chain"
            );

            registered_job_id
        };

        info!(
            job_id = job_id,
            elapsed_ms = phase5_start.elapsed().as_millis(),
            "Phase 5 complete: training job registered"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 5, elapsed_ms: phase5_start.elapsed().as_millis() });

        // -- Phase 5.5: Generate and fund worker wallets if needed --
        // On remote testnets (not Anvil), workers don't come pre-funded.
        // Generate random wallets and transfer funds from the owner.
        if self.config.worker_private_keys.is_empty() && num_workers > 0 {
            info!(
                "Phase 5.5: No worker keys provided — generating {} random wallets",
                num_workers
            );
            let provider = ethers::providers::Provider::<ethers::providers::Http>::try_from(&rpc_url)
                .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;
            let chain_id = provider.get_chainid().await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?.as_u64();

            let owner_pk = self.config.private_key.strip_prefix("0x")
                .unwrap_or(&self.config.private_key);
            let owner_wallet = LocalWallet::from_str(owner_pk)
                .map_err(|e| anyhow!("Invalid owner private key: {}", e))?
                .with_chain_id(chain_id);
            let owner_signer = ethers::middleware::SignerMiddleware::new(
                provider.clone(),
                owner_wallet.clone(),
            );

            // Each worker needs stake + gas. Fund with 2x stake to cover gas.
            let stake_wei = ethers::utils::parse_ether(self.config.stake_amount_eth)
                .context("Invalid stake amount")?;
            let fund_amount = stake_wei * 3; // stake + generous gas buffer

            for i in 0..num_workers {
                let worker_wallet = LocalWallet::new(&mut rand::thread_rng());
                let worker_key = format!("0x{}", hex::encode(worker_wallet.signer().to_bytes()));

                let tx = ethers::types::TransactionRequest::new()
                    .to(worker_wallet.address())
                    .value(fund_amount);
                let pending = owner_signer.send_transaction(tx, None).await
                    .with_context(|| format!("Phase 5.5: Failed to fund worker {}", i))?;
                let _receipt = pending.await
                    .with_context(|| format!("Phase 5.5: Worker {} funding tx not confirmed", i))?;

                info!(
                    worker = i,
                    address = %worker_wallet.address(),
                    "Funded worker wallet"
                );
                self.config.worker_private_keys.push(worker_key);
            }
        }

        // -- Phase 6: Workers stake and join --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 6, total: 13,
            description: "Workers staking and joining".to_string(),
        });
        info!("Phase 6: Workers staking and joining");
        let phase6_start = Instant::now();

        let mut worker_wallets = Vec::with_capacity(num_workers);

        if self.config.use_pool_workers {
            // ── On-chain pool assignment ──
            // Workers have already registered in the global pool via `registerInPool()`.
            // The owner simply assigns available pool workers to this job.
            info!(
                "Phase 6: Using on-chain worker pool — assigning {} workers to job {}",
                num_workers, job_id
            );

            let assign_receipt = chain_client
                .assign_pool_workers(job_id, num_workers as u64)
                .await
                .context("Phase 6: assign_pool_workers failed — are enough workers registered in the pool?")?;

            let assign_gas = assign_receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += assign_gas;

            info!(
                workers_assigned = num_workers,
                gas_used = assign_gas,
                "Pool workers assigned on-chain"
            );

            // Parse worker wallets for demo-mode attestation signing.
            // In production, workers sign their own attestations via their nodes.
            let provider = ethers::providers::Provider::<ethers::providers::Http>::try_from(&rpc_url)
                .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;
            let chain_id = provider
                .get_chainid()
                .await
                .map_err(|e| anyhow!("Failed to get chain ID: {}", e))?
                .as_u64();

            for (i, wk_key) in self.config.worker_private_keys.iter().enumerate() {
                let pk = wk_key.strip_prefix("0x").unwrap_or(wk_key);
                let wallet = LocalWallet::from_str(pk)
                    .map_err(|e| anyhow!("Invalid worker {} private key: {}", i, e))?
                    .with_chain_id(chain_id);
                worker_wallets.push(wallet);
            }
        } else {
            // ── Legacy per-worker staking ──
            let stake_wei = ethers::utils::parse_ether(self.config.stake_amount_eth)
                .context("Phase 6: Invalid stake amount")?;

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
        }

        info!(
            workers_staked = worker_wallets.len(),
            elapsed_ms = phase6_start.elapsed().as_millis(),
            "Phase 6 complete: all workers staked"
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 6, elapsed_ms: phase6_start.elapsed().as_millis() });

        // -- Phase 7: Poll for worker readiness --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 7, total: 13,
            description: "Waiting for worker readiness".to_string(),
        });
        info!("Phase 7: Waiting for all workers to be registered on-chain");
        let phase7_start = Instant::now();
        let expected_workers = num_workers as u64;
        let poll_timeout = Duration::from_secs(60);
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
                    "Phase 7: Timed out waiting for {} workers after {}s. Only {} registered. \
                     Ensure all workers are running and can reach the coordinator at {}.",
                    expected_workers,
                    poll_timeout.as_secs(),
                    count,
                    coordinator_addr_str
                ));
            }

            tokio::time::sleep(poll_interval).await;
        }
        self.emit(ProgressEvent::PhaseCompleted { phase: 7, elapsed_ms: phase7_start.elapsed().as_millis() });

        Ok((job_id, coordinator_addr_str, chain_client, worker_wallets, total_gas))
    }

    // ========================================================================
    // Phases 9-11: On-chain settlement (feature-gated)
    // ========================================================================

    /// Runs phases 9 through 11: checkpoint attestation, cheater slashing,
    /// and training completion.
    ///
    /// Returns (checkpoints_on_chain, cheater_info, gas_used, zk_proofs_on_chain).
    #[cfg(feature = "chain")]
    async fn run_chain_settlement_phases(
        &mut self,
        job_id: u64,
        mpc_result: &MPCIntegrationResult,
        chain_client: &ChainClientV4,
        worker_wallets: &[LocalWallet],
    ) -> Result<(usize, Option<CheaterInfo>, u64, usize)> {
        let mut total_gas: u64 = 0;
        let mut checkpoints_on_chain: usize = 0;
        let mut zk_proofs_on_chain: usize = 0;

        // -- Phase 9: Submit checkpoints (with optional ZK proofs) --
        let zk_enabled = self.zk_layer.is_some();
        self.emit(ProgressEvent::PhaseStarted {
            phase: 9, total: 13,
            description: if zk_enabled {
                "Submitting checkpoint attestations + ZK proofs".to_string()
            } else {
                "Submitting checkpoint attestations".to_string()
            },
        });
        info!(
            checkpoint_count = mpc_result.checkpoints.len(),
            zk_enabled = zk_enabled,
            "Phase 9: Submitting checkpoint attestations"
        );
        let phase9_start = Instant::now();
        let total_checkpoints = mpc_result.checkpoints.len();

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

            // Optionally generate a ZK proof for the LAST checkpoint.
            // In MPC training, intermediate checkpoint weights aren't reconstructed
            // (only commitments are recorded). We have initial weights (baseline,
            // set in Phase 2) and final weights (reconstructed after MPC completes).
            // The ZK proof covers the transition: initial_weights → final_weights.
            let is_last_checkpoint = idx == total_checkpoints - 1;
            let zk_proof_result: Option<ZkCheckpointProofResult> = if is_last_checkpoint {
                if let Some(ref mut zk_layer) = self.zk_layer {
                    self.progress.as_ref().map(|cb| cb(ProgressEvent::ZkProofStarted {
                        checkpoint_index: idx,
                        step: checkpoint.step as u64,
                    }));

                    match zk_layer.process_final_checkpoint(
                        idx,
                        checkpoint.step,
                        &mpc_result.final_weights,
                        checkpoint.loss,
                    ) {
                        Ok(Some(result)) => {
                            if let Some(ref cb) = self.progress {
                                cb(ProgressEvent::ZkProofGenerated {
                                    checkpoint_index: idx,
                                    step: checkpoint.step as u64,
                                    proof_size: result.proof.proof_size(),
                                    time_ms: result.proof.generation_time.as_millis() as u64,
                                    verified: result.proof.verified,
                                });
                            }
                            Some(result)
                        }
                        Ok(None) => None,
                        Err(e) => {
                            // ZK proof failure is non-fatal; MAC attestation is primary
                            warn!(
                                checkpoint_index = idx,
                                error = %e,
                                "Phase 9: ZK proof generation failed (non-fatal), falling back to attestation-only"
                            );
                            if let Some(ref cb) = self.progress {
                                cb(ProgressEvent::ZkProofFailed {
                                    checkpoint_index: idx,
                                    step: checkpoint.step as u64,
                                    error: e.to_string(),
                                });
                            }
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Submit checkpoint: use ZK path if we have a proof, otherwise attestation path.
            let receipt = if let Some(ref zk_result) = zk_proof_result {
                let evm_proof = zk_result.proof.to_evm_proof();
                let evm_public_inputs: Vec<U256> = zk_result
                    .proof
                    .to_evm_public_inputs()
                    .iter()
                    .map(|bytes| U256::from_big_endian(bytes))
                    .collect();

                info!(
                    checkpoint_index = idx,
                    proof_size = evm_proof.len(),
                    public_inputs = evm_public_inputs.len(),
                    "Submitting checkpoint with ZK proof"
                );

                retry_chain_op(
                    &format!("checkpoint {} ZK submission", idx),
                    || {
                        let proof_clone = evm_proof.clone();
                        let pi_clone = evm_public_inputs.clone();
                        async move {
                            chain_client
                                .submit_checkpoint_with_proof(
                                    job_id,
                                    checkpoint.step as u64,
                                    checkpoint.commitment_bytes32,
                                    loss_u256,
                                    proof_clone,
                                    pi_clone,
                                )
                                .await
                        }
                    },
                    &self.progress,
                    9,
                ).await
                .with_context(|| format!("Phase 9: Failed to submit ZK checkpoint {} after retries", idx))?
            } else {
                retry_chain_op(
                    &format!("checkpoint {} submission", idx),
                    || async {
                        chain_client
                            .submit_checkpoint(
                                job_id,
                                checkpoint.step as u64,
                                checkpoint.commitment_bytes32,
                                loss_u256,
                                signatures.clone(),
                            )
                            .await
                    },
                    &self.progress,
                    9,
                ).await
                .with_context(|| format!("Phase 9: Failed to submit checkpoint {} after retries", idx))?
            };

            let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += gas;
            checkpoints_on_chain += 1;

            if zk_proof_result.is_some() {
                zk_proofs_on_chain += 1;
                let tx_hash = format!("{:?}", receipt.transaction_hash);
                if let Some(ref cb) = self.progress {
                    cb(ProgressEvent::ZkProofSubmitted {
                        checkpoint_index: idx,
                        step: checkpoint.step as u64,
                        tx_hash: tx_hash.clone(),
                    });
                }
            }

            let tx_hash = format!("{:?}", receipt.transaction_hash);
            self.emit(ProgressEvent::CheckpointSubmitted {
                index: idx + 1,
                total: total_checkpoints,
                step: checkpoint.step as u64,
                tx_hash: tx_hash.clone(),
            });

            debug!(
                checkpoint_index = idx,
                step = checkpoint.step,
                loss = checkpoint.loss,
                gas_used = gas,
                zk_proof = zk_proof_result.is_some(),
                tx_hash = %tx_hash,
                "Checkpoint submitted on-chain"
            );
        }

        let zk_stats_msg = if let Some(ref zk_layer) = self.zk_layer {
            let stats = zk_layer.stats();
            format!(
                ", zk_proofs={}, zk_on_chain={}, zk_proving_time_ms={}",
                stats.proofs_generated,
                zk_proofs_on_chain,
                stats.total_proving_time.as_millis()
            )
        } else {
            String::new()
        };

        info!(
            checkpoints_submitted = checkpoints_on_chain,
            elapsed_ms = phase9_start.elapsed().as_millis(),
            "Phase 9 complete: all checkpoints attested{}",
            zk_stats_msg
        );
        self.emit(ProgressEvent::PhaseCompleted { phase: 9, elapsed_ms: phase9_start.elapsed().as_millis() });

        // -- Phase 10: Report cheater if detected --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 10, total: 13,
            description: if mpc_result.cheater_detected.is_some() {
                "Reporting MAC failure on-chain".to_string()
            } else {
                "Checking for cheaters (none detected)".to_string()
            },
        });
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
                    self.emit(ProgressEvent::CheaterSlashed {
                        party_index: cheater.party_index,
                        tx_hash: tx_hash.clone(),
                    });
                    (true, Some(tx_hash))
                }
                Err(e) => {
                    error!("Phase 10: MAC failure report failed: {}", e);
                    self.emit(ProgressEvent::RecoverableError {
                        phase: 10,
                        message: format!("MAC failure report failed: {}", e),
                        retry_count: 0,
                    });
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
        self.emit(ProgressEvent::PhaseCompleted { phase: 10, elapsed_ms: 0 });

        // -- Phase 10.5: Risk-based ZK activation check --
        // If zk_mode is Risk and a cheater was just slashed, the active worker
        // count may have dropped below the threshold. Query the contract to see
        // if ZK is now required, and dynamically create a ZkProofLayer if so.
        if matches!(self.config.zk_mode, ZkMode::Risk { .. }) && self.zk_layer.is_none() {
            match chain_client.is_zk_required(job_id).await {
                Ok(true) => {
                    let active_workers = chain_client
                        .get_active_worker_count(job_id)
                        .await
                        .unwrap_or(0);
                    let min_w = match self.config.zk_mode {
                        ZkMode::Risk { min_workers } => min_workers,
                        _ => 0,
                    };

                    warn!(
                        active_workers = active_workers,
                        min_workers = min_w,
                        "Phase 10.5: Risk threshold crossed! Active workers ({}) < min_workers ({}). \
                         Activating ZK proof layer for remaining operations.",
                        active_workers,
                        min_w,
                    );

                    self.emit(ProgressEvent::ZkRiskActivated {
                        active_workers,
                        min_workers: min_w,
                    });

                    // Dynamically create the ZK proof layer.
                    let arch = &self.config.architecture;
                    if arch.len() == 3 {
                        let d_in = arch[0];
                        let d_hid = arch[1];
                        let d_out = arch[2];

                        let mut zk_config = self.config.zk_proof.clone();
                        zk_config.enabled = true; // Force-enable for risk activation.

                        let mut zk_layer = ZkProofLayer::new(zk_config, d_in, d_hid, d_out);

                        // Set baseline weights from the final MPC result so the proof
                        // covers the full training transition.
                        if let Err(e) = zk_layer.set_baseline_weights(
                            &mpc_result.final_weights.w1,
                            &mpc_result.final_weights.b1,
                            &mpc_result.final_weights.w2,
                            &mpc_result.final_weights.b2,
                        ) {
                            warn!(
                                error = %e,
                                "Phase 10.5: Failed to set baseline weights for risk-activated ZK layer"
                            );
                        } else {
                            info!("Phase 10.5: ZK proof layer activated dynamically for risk mode");
                            self.zk_layer = Some(zk_layer);
                        }
                    } else {
                        warn!(
                            "Phase 10.5: Cannot activate ZK layer — architecture must have 3 layers, got {}",
                            arch.len()
                        );
                    }
                }
                Ok(false) => {
                    debug!("Phase 10.5: ZK not required (worker count still above threshold)");
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "Phase 10.5: Failed to query is_zk_required, continuing without ZK activation"
                    );
                }
            }
        }

        // -- Phase 11: Complete training on-chain --
        self.emit(ProgressEvent::PhaseStarted {
            phase: 11, total: 13,
            description: "Completing training on-chain".to_string(),
        });
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
        self.emit(ProgressEvent::PhaseCompleted { phase: 11, elapsed_ms: phase11_start.elapsed().as_millis() });

        Ok((checkpoints_on_chain, cheater_info, total_gas, zk_proofs_on_chain))
    }

    // ========================================================================
    // Phase 11.5: Stake Withdrawal (feature-gated)
    // ========================================================================

    /// Runs the stake withdrawal phase: advances time past the 7-day cooldown
    /// (local Anvil only) and has each honest worker withdraw their stake.
    ///
    /// Returns gas used across all withdrawals.
    #[cfg(feature = "chain")]
    async fn run_withdrawal_phase(
        &self,
        job_id: u64,
        coordinator_address: &str,
        worker_wallets: &[LocalWallet],
        cheater_info: &Option<CheaterInfo>,
    ) -> Result<u64> {
        info!("Phase 11.5: Withdrawing stakes after cooldown");
        let phase_start = Instant::now();

        let rpc_url = self
            .rpc_url
            .as_ref()
            .ok_or_else(|| anyhow!("Phase 11.5: No RPC URL available"))?;

        // Advance time past the 7-day cooldown (only works on local Anvil).
        let provider = Provider::<Http>::try_from(rpc_url.as_str())
            .map_err(|e| anyhow!("Invalid RPC URL: {}", e))?;

        let seven_days_plus: u64 = 7 * 24 * 3600 + 1;
        let _: serde_json::Value = provider
            .request("evm_increaseTime", [seven_days_plus])
            .await
            .map_err(|e| anyhow!("Phase 11.5: evm_increaseTime failed (not on Anvil?): {}", e))?;
        let _: serde_json::Value = provider
            .request("evm_mine", Vec::<()>::new())
            .await
            .map_err(|e| anyhow!("Phase 11.5: evm_mine failed: {}", e))?;

        info!("Advanced time by 7 days + 1 second past cooldown");

        // Withdraw for each honest worker.
        let mut total_gas: u64 = 0;
        let mut withdrawals: usize = 0;

        for (i, wallet) in worker_wallets.iter().enumerate() {
            // Skip the cheater — their stake was already slashed.
            if let Some(ref ci) = cheater_info {
                if i == ci.party_index {
                    debug!(worker = i, "Skipping slashed worker for withdrawal");
                    continue;
                }
            }

            let worker_client = ChainClientV4::with_wallet(
                rpc_url,
                wallet.clone(),
                coordinator_address,
            )
            .await
            .with_context(|| format!("Phase 11.5: Failed to create worker {} client", i))?;

            let receipt = worker_client
                .withdraw_stake(job_id)
                .await
                .with_context(|| format!("Phase 11.5: Worker {} withdraw_stake failed", i))?;

            let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
            total_gas += gas;
            withdrawals += 1;

            debug!(
                worker = i,
                address = %wallet.address(),
                gas_used = gas,
                "Worker stake withdrawn"
            );
        }

        info!(
            withdrawals = withdrawals,
            total_gas = total_gas,
            elapsed_ms = phase_start.elapsed().as_millis(),
            "Phase 11.5 complete: stakes withdrawn"
        );

        Ok(total_gas)
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

        if result.zk_proofs_generated > 0 {
            info!("");
            info!("  ZK Proof Layer");
            info!("  {thin_divider}");
            info!("    Proofs Generated: {}", result.zk_proofs_generated);
            info!("    Proofs On-Chain:  {}", result.zk_proofs_on_chain);
            if let Some(ref zk_layer) = self.zk_layer {
                let stats = zk_layer.stats();
                info!("    Proofs Verified:  {}", stats.proofs_verified);
                info!("    Proving Time:     {:.2}s", stats.total_proving_time.as_secs_f64());
                if let Some(init_time) = stats.init_time {
                    info!("    Init Time:        {:.2}s", init_time.as_secs_f64());
                }
            }
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

/// Retries a chain operation up to `CHAIN_RETRY_MAX` times with exponential backoff.
///
/// On each retry, emits a `RecoverableError` progress event so the CLI can
/// inform the user. Returns the result of the last attempt on final failure.
#[cfg(feature = "chain")]
async fn retry_chain_op<F, Fut, T>(
    operation_name: &str,
    mut op: F,
    progress: &Option<ProgressCallback>,
    phase: u32,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_err = None;
    for attempt in 0..CHAIN_RETRY_MAX {
        match op().await {
            Ok(val) => return Ok(val),
            Err(e) => {
                let msg = format!(
                    "{} failed (attempt {}/{}): {}",
                    operation_name,
                    attempt + 1,
                    CHAIN_RETRY_MAX,
                    e
                );
                warn!("{}", msg);
                if let Some(ref cb) = progress {
                    cb(ProgressEvent::RecoverableError {
                        phase,
                        message: msg,
                        retry_count: attempt + 1,
                    });
                }
                last_err = Some(e);
                if attempt + 1 < CHAIN_RETRY_MAX {
                    let delay = CHAIN_RETRY_DELAY * 2u32.pow(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("{} failed after {} retries", operation_name, CHAIN_RETRY_MAX)))
}

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
        assert_eq!(config.architecture, vec![784, 128, 10]);
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
            batch_size: 1,
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
            payment_amount_eth: 0.01,
            #[cfg(feature = "chain")]
            stake_amount_eth: 0.001,
            #[cfg(feature = "chain")]
            coordinator_address: None,
            #[cfg(feature = "chain")]
            enable_withdrawal: false,
            #[cfg(feature = "chain")]
            use_pool_workers: false,
            #[cfg(feature = "chain")]
            pre_registered_job_id: None,
            #[cfg(feature = "chain")]
            model_store_address: None,
            #[cfg(feature = "chain")]
            model_slug: None,
            #[cfg(feature = "chain")]
            model_name: None,
            #[cfg(feature = "chain")]
            model_description: None,
            distributed: false,
            zk_proof: ZkProofConfig::default(),
            zk_mode: ZkMode::Off,
            custom_training_data: None,
            simulate_cheater: false,
            cheater_party: None,
            cheater_step: None,
            worker_seeds: Vec::new(),
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
            zk_proofs_generated: 0,
            zk_proofs_on_chain: 0,
            model_nft_token_id: None,
            model_store_address: String::new(),
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
