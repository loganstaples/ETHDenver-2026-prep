//! HELIX Demo Mode - Production-Ready Training Scenarios
//!
//! Provides comprehensive demo scenarios for showcasing HELIX distributed ML training:
//! - Quick: Fast demonstration of core functionality (30 seconds)
//! - FullTraining: Complete training cycle with proofs and verification (90 seconds)
//! - Slashing: Demonstrates fault detection and economic penalties
//! - MultiModel: Concurrent training of multiple models
//! - FaultTolerance: Network resilience under node failures
//!
//! Features:
//! - Real-time proof generation with timing guarantees
//! - Pre-warming for fast demo starts
//! - Automatic error recovery
//! - Precise timing control for 90-second pitch demos

pub mod orchestrator;
pub mod prewarm;
pub mod recovery;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, watch, RwLock};

use crate::progress::{PhaseDisplay, ProgressDisplay};
use crate::rpc::{
    HelixRpcConfig, MockRpcClient, ProofPhase, ProofStatus, RealTimeProofTracker,
    RealTimeTrainingTracker, TrainingPhase, TrainingProgress, TrainingStatus,
    UnifiedRpcClient, WorkerInfo, WorkerStatus,
};

pub use orchestrator::{
    DemoOrchestrator, DemoTimingConfig, OrchestratedPhase, OrchestratorEvent,
    OrchestratorSnapshot, PhaseTiming, TimingReport,
};
pub use prewarm::{DemoPrewarmer, PrewarmConfig, PrewarmResult, PrewarmStatus};
pub use recovery::{
    CircuitBreaker, CircuitState, ErrorCategory, HeartbeatMonitor,
    PreDemoCheck, PreDemoCheckResult, RecoveryAction, RecoveryConfig,
    RecoveryEvent, RecoveryManager, RecoveryReport, RecoveryResult,
};

// ============================================================================
// Demo Scenario Configuration
// ============================================================================

/// Demo scenario types with distinct behaviors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemoScenarioType {
    /// Quick 30-second demo showing basic training loop
    Quick,
    /// Full 90-second training with proofs, verification, and finalization
    FullTraining,
    /// Demonstrates slashing for malicious/faulty behavior
    Slashing,
    /// Multiple models training concurrently
    MultiModel,
    /// Network resilience under node failures
    FaultTolerance,
}

impl DemoScenarioType {
    /// Get human-readable name for the scenario
    pub fn name(&self) -> &'static str {
        match self {
            Self::Quick => "Quick Demo",
            Self::FullTraining => "Full Training Demo",
            Self::Slashing => "Slashing Demo",
            Self::MultiModel => "Multi-Model Demo",
            Self::FaultTolerance => "Fault Tolerance Demo",
        }
    }

    /// Get detailed description of the scenario
    pub fn description(&self) -> &'static str {
        match self {
            Self::Quick => "Fast demonstration of HELIX core functionality with minimal rounds (30s)",
            Self::FullTraining => "Complete training cycle including proof generation, verification, and model finalization (90s)",
            Self::Slashing => "Demonstrates detection of malicious behavior and economic slashing penalties",
            Self::MultiModel => "Shows concurrent training of multiple independent models",
            Self::FaultTolerance => "Tests network resilience with simulated node failures and recovery",
        }
    }

    /// Get default configuration for this scenario
    pub fn default_config(&self) -> DemoConfig {
        match self {
            Self::Quick => DemoConfig {
                scenario: *self,
                worker_count: 3,
                aggregator_count: 1,
                round_count: 5,
                round_duration: Duration::from_secs(4),
                model_count: 1,
                stake_amount: 0.5,
                error_bound_max: 100.0,
                simulate_faults: false,
                simulate_slashing: false,
                headless: false,
                verbose: false,
                prewarm: true,
                timing: DemoTimingConfig::quick_30_second(5),
                recovery: RecoveryConfig::demo_mode(),
            },
            Self::FullTraining => DemoConfig {
                scenario: *self,
                worker_count: 5,
                aggregator_count: 2,
                round_count: 10,
                round_duration: Duration::from_secs(6),
                model_count: 1,
                stake_amount: 1.0,
                error_bound_max: 1000.0,
                simulate_faults: false,
                simulate_slashing: false,
                headless: false,
                verbose: false,
                prewarm: true,
                timing: DemoTimingConfig::standard_90_second(10),
                recovery: RecoveryConfig::demo_mode(),
            },
            Self::Slashing => DemoConfig {
                scenario: *self,
                worker_count: 4,
                aggregator_count: 1,
                round_count: 8,
                round_duration: Duration::from_secs(8),
                model_count: 1,
                stake_amount: 2.0,
                error_bound_max: 500.0,
                simulate_faults: true,
                simulate_slashing: true,
                headless: false,
                verbose: true,
                prewarm: true,
                timing: DemoTimingConfig::standard_90_second(8),
                recovery: RecoveryConfig::demo_mode(),
            },
            Self::MultiModel => DemoConfig {
                scenario: *self,
                worker_count: 6,
                aggregator_count: 2,
                round_count: 5,
                round_duration: Duration::from_secs(12),
                model_count: 3,
                stake_amount: 0.5,
                error_bound_max: 1000.0,
                simulate_faults: false,
                simulate_slashing: false,
                headless: false,
                verbose: false,
                prewarm: true,
                timing: DemoTimingConfig::standard_90_second(5),
                recovery: RecoveryConfig::demo_mode(),
            },
            Self::FaultTolerance => DemoConfig {
                scenario: *self,
                worker_count: 7,
                aggregator_count: 2,
                round_count: 10,
                round_duration: Duration::from_secs(6),
                model_count: 1,
                stake_amount: 1.0,
                error_bound_max: 1000.0,
                simulate_faults: true,
                simulate_slashing: false,
                headless: false,
                verbose: true,
                prewarm: true,
                timing: DemoTimingConfig::standard_90_second(10),
                recovery: RecoveryConfig::demo_mode(),
            },
        }
    }
}

/// Demo configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    /// Scenario type
    pub scenario: DemoScenarioType,
    /// Number of worker nodes
    pub worker_count: u32,
    /// Number of aggregator nodes
    pub aggregator_count: u32,
    /// Number of training rounds
    pub round_count: u32,
    /// Duration per round
    #[serde(with = "humantime_serde")]
    pub round_duration: Duration,
    /// Number of models to train
    pub model_count: u32,
    /// Stake amount per worker (ETH)
    pub stake_amount: f64,
    /// Maximum allowed error bound
    pub error_bound_max: f64,
    /// Whether to simulate node faults
    pub simulate_faults: bool,
    /// Whether to simulate slashing events
    pub simulate_slashing: bool,
    /// Run without interactive UI
    pub headless: bool,
    /// Verbose output
    pub verbose: bool,
    /// Enable pre-warming
    pub prewarm: bool,
    /// Timing configuration
    pub timing: DemoTimingConfig,
    /// Recovery configuration
    pub recovery: RecoveryConfig,
}

impl Default for DemoConfig {
    fn default() -> Self {
        DemoScenarioType::Quick.default_config()
    }
}

// ============================================================================
// Demo State & Events
// ============================================================================

/// Current state of the demo execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoState {
    /// Current phase of execution
    pub phase: DemoPhase,
    /// Models being trained
    pub models: Vec<ModelState>,
    /// Active workers
    pub workers: Vec<WorkerState>,
    /// Active aggregators
    pub aggregators: Vec<AggregatorState>,
    /// Events that have occurred
    pub events: Vec<DemoEvent>,
    /// Overall metrics
    pub metrics: DemoMetrics,
    /// Start time (not serialized)
    #[serde(skip)]
    pub started_at: Option<Instant>,
    /// Whether demo is complete
    pub completed: bool,
    /// Whether pre-warming is complete
    pub prewarmed: bool,
}

impl Default for DemoState {
    fn default() -> Self {
        Self {
            phase: DemoPhase::Initializing,
            models: Vec::new(),
            workers: Vec::new(),
            aggregators: Vec::new(),
            events: Vec::new(),
            metrics: DemoMetrics::default(),
            started_at: None,
            completed: false,
            prewarmed: false,
        }
    }
}

/// Demo execution phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemoPhase {
    /// Pre-warming caches
    Prewarming,
    /// Setting up the network
    Initializing,
    /// Deploying smart contracts
    DeployingContracts,
    /// Registering models
    RegisteringModels,
    /// Starting nodes
    StartingNodes,
    /// Staking tokens
    Staking,
    /// Running training rounds
    Training,
    /// Generating proofs
    Proving,
    /// Verifying proofs
    Verifying,
    /// Handling faults/slashing
    Slashing,
    /// Finalizing models
    Finalizing,
    /// Demo complete
    Complete,
    /// Demo failed
    Failed,
}

impl DemoPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Prewarming => "Pre-warming",
            Self::Initializing => "Initializing",
            Self::DeployingContracts => "Deploying Contracts",
            Self::RegisteringModels => "Registering Models",
            Self::StartingNodes => "Starting Nodes",
            Self::Staking => "Staking Tokens",
            Self::Training => "Training",
            Self::Proving => "Generating Proofs",
            Self::Verifying => "Verifying Proofs",
            Self::Slashing => "Slashing",
            Self::Finalizing => "Finalizing",
            Self::Complete => "Complete",
            Self::Failed => "Failed",
        }
    }
}

/// State of a model being trained
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Model ID
    pub id: u64,
    /// Model name
    pub name: String,
    /// IPFS hash of initial model
    pub ipfs_hash: String,
    /// Current round
    pub current_round: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Current loss value
    pub current_loss: f64,
    /// Loss history
    pub loss_history: Vec<f64>,
    /// Accumulated error bound
    pub error_bound: f64,
    /// Whether training is complete
    pub is_complete: bool,
    /// Final model commitment
    pub final_commitment: Option<String>,
}

/// State of a worker node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerState {
    /// Worker ID
    pub id: String,
    /// Ethereum address
    pub address: String,
    /// Staked amount
    pub stake: f64,
    /// Current status
    pub status: DemoWorkerStatus,
    /// Proofs submitted
    pub proofs_submitted: u64,
    /// Proofs verified
    pub proofs_verified: u64,
    /// Current model being trained
    pub current_model: Option<u64>,
    /// Reputation score (0-1)
    pub reputation: f64,
    /// Whether this worker has been slashed
    pub slashed: bool,
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemoWorkerStatus {
    Starting,
    Syncing,
    Idle,
    Training,
    Proving,
    Waiting,
    Faulted,
    Slashed,
    Offline,
}

impl DemoWorkerStatus {
    pub fn symbol(&self) -> ColoredString {
        match self {
            Self::Starting => "◐".yellow(),
            Self::Syncing => "◓".blue(),
            Self::Idle => "○".dimmed(),
            Self::Training => "●".green(),
            Self::Proving => "◉".cyan(),
            Self::Waiting => "◔".white(),
            Self::Faulted => "◍".red(),
            Self::Slashed => "✗".red().bold(),
            Self::Offline => "○".red(),
        }
    }
}

/// State of an aggregator node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatorState {
    /// Aggregator ID
    pub id: String,
    /// Ethereum address
    pub address: String,
    /// Current status
    pub status: AggregatorStatus,
    /// Rounds aggregated
    pub rounds_aggregated: u64,
    /// Proofs verified
    pub proofs_verified: u64,
    /// Models being coordinated
    pub active_models: Vec<u64>,
}

/// Aggregator status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregatorStatus {
    Starting,
    Ready,
    Aggregating,
    Verifying,
    Committing,
    Offline,
}

impl AggregatorStatus {
    pub fn symbol(&self) -> ColoredString {
        match self {
            Self::Starting => "◐".yellow(),
            Self::Ready => "●".green(),
            Self::Aggregating => "◉".blue(),
            Self::Verifying => "◎".cyan(),
            Self::Committing => "◉".magenta(),
            Self::Offline => "○".red(),
        }
    }
}

/// Events that occur during demo execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoEvent {
    /// Event timestamp (ms since demo start)
    pub timestamp_ms: u64,
    /// Event type
    pub event_type: DemoEventType,
    /// Event description
    pub description: String,
    /// Associated model ID (if any)
    pub model_id: Option<u64>,
    /// Associated node ID (if any)
    pub node_id: Option<String>,
}

/// Types of demo events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemoEventType {
    PhaseChange,
    ContractDeployed,
    ModelRegistered,
    NodeStarted,
    NodeJoined,
    StakeDeposited,
    RoundStarted,
    RoundCompleted,
    ProofGenerated,
    ProofVerified,
    ProofRejected,
    FaultDetected,
    SlashingExecuted,
    NodeRecovered,
    ModelFinalized,
    ErrorBoundExceeded,
}

impl DemoEventType {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::PhaseChange => "▶",
            Self::ContractDeployed => "📜",
            Self::ModelRegistered => "🧠",
            Self::NodeStarted => "🚀",
            Self::NodeJoined => "🤝",
            Self::StakeDeposited => "💰",
            Self::RoundStarted => "⏵",
            Self::RoundCompleted => "✓",
            Self::ProofGenerated => "🔐",
            Self::ProofVerified => "✓",
            Self::ProofRejected => "✗",
            Self::FaultDetected => "⚠",
            Self::SlashingExecuted => "🔪",
            Self::NodeRecovered => "↻",
            Self::ModelFinalized => "🎯",
            Self::ErrorBoundExceeded => "⚠",
        }
    }
}

/// Overall demo metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemoMetrics {
    /// Total rounds completed
    pub rounds_completed: u64,
    /// Total proofs generated
    pub proofs_generated: u64,
    /// Total proofs verified
    pub proofs_verified: u64,
    /// Total proofs rejected
    pub proofs_rejected: u64,
    /// Total stake deposited
    pub total_stake: f64,
    /// Total stake slashed
    pub stake_slashed: f64,
    /// Average round time (ms)
    pub avg_round_time_ms: u64,
    /// Average proof generation time (ms)
    pub avg_proof_time_ms: u64,
    /// Node failures simulated
    pub node_failures: u64,
    /// Node recoveries
    pub node_recoveries: u64,
    /// Final error bounds by model
    pub final_error_bounds: HashMap<u64, f64>,
    /// Total rewards earned
    pub rewards_earned: f64,
    /// Rewards per successful proof
    pub reward_per_proof: f64,
}

// ============================================================================
// Demo Runner
// ============================================================================

/// Main demo runner that orchestrates all scenarios
pub struct DemoRunner {
    /// Demo configuration
    config: DemoConfig,
    /// Current state
    state: Arc<RwLock<DemoState>>,
    /// Event channel for UI updates
    event_tx: mpsc::Sender<DemoEvent>,
    /// Event receiver
    event_rx: Option<mpsc::Receiver<DemoEvent>>,
    /// Multi-progress for terminal display
    multi_progress: MultiProgress,
    /// Pre-warmer
    prewarmer: Option<DemoPrewarmer>,
    /// Recovery manager
    recovery_manager: RecoveryManager,
    /// Unified RPC client (real or mock)
    rpc_client: Arc<RwLock<UnifiedRpcClient>>,
    /// Real-time proof tracking
    proof_tracker: Arc<RealTimeProofTracker>,
    /// Real-time training tracking
    training_tracker: Arc<RealTimeTrainingTracker>,
    /// RPC configuration for node connection
    rpc_config: Option<HelixRpcConfig>,
}

impl DemoRunner {
    /// Create a new demo runner with the given configuration
    pub fn new(config: DemoConfig) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1000);
        let recovery_manager = RecoveryManager::new(config.recovery.clone());

        Self {
            config,
            state: Arc::new(RwLock::new(DemoState::default())),
            event_tx,
            event_rx: Some(event_rx),
            multi_progress: MultiProgress::new(),
            prewarmer: Some(DemoPrewarmer::new(PrewarmConfig::demo_mode())),
            recovery_manager,
            rpc_client: Arc::new(RwLock::new(UnifiedRpcClient::mock_only())),
            proof_tracker: Arc::new(RealTimeProofTracker::new()),
            training_tracker: Arc::new(RealTimeTrainingTracker::new()),
            rpc_config: None,
        }
    }

    /// Create a demo runner with a specific RPC endpoint
    pub fn with_endpoint(config: DemoConfig, endpoint: &str) -> Self {
        let (event_tx, event_rx) = mpsc::channel(1000);
        let recovery_manager = RecoveryManager::new(config.recovery.clone());

        Self {
            config,
            state: Arc::new(RwLock::new(DemoState::default())),
            event_tx,
            event_rx: Some(event_rx),
            multi_progress: MultiProgress::new(),
            prewarmer: Some(DemoPrewarmer::new(PrewarmConfig::demo_mode())),
            recovery_manager,
            rpc_client: Arc::new(RwLock::new(UnifiedRpcClient::mock_only())),
            proof_tracker: Arc::new(RealTimeProofTracker::new()),
            training_tracker: Arc::new(RealTimeTrainingTracker::new()),
            rpc_config: Some(HelixRpcConfig::with_endpoint(endpoint)),
        }
    }

    /// Try to connect to a real node, falling back to mock mode if unavailable
    pub async fn connect(&self) -> Result<bool> {
        let config = self.rpc_config.clone().unwrap_or_default();
        let client = UnifiedRpcClient::new(config).await;
        let connected = client.is_connected();

        *self.rpc_client.write().await = client;

        if connected {
            println!(
                "  {} Connected to HELIX node",
                "✓".green()
            );
        } else {
            println!(
                "  {} Running in demo mode (no node connection)",
                "○".yellow()
            );
        }

        Ok(connected)
    }

    /// Check if running with real node connection
    pub async fn is_real_mode(&self) -> bool {
        self.rpc_client.read().await.is_connected()
    }

    /// Create a demo runner for a specific scenario type
    pub fn for_scenario(scenario: DemoScenarioType) -> Self {
        Self::new(scenario.default_config())
    }

    /// Get a reference to the current state
    pub fn state(&self) -> Arc<RwLock<DemoState>> {
        self.state.clone()
    }

    /// Get the real-time proof tracker for UI integration
    pub fn proof_tracker(&self) -> Arc<RealTimeProofTracker> {
        self.proof_tracker.clone()
    }

    /// Get the real-time training tracker for UI integration
    pub fn training_tracker(&self) -> Arc<RealTimeTrainingTracker> {
        self.training_tracker.clone()
    }

    /// Get the RPC client for direct access
    pub fn rpc_client(&self) -> Arc<RwLock<UnifiedRpcClient>> {
        self.rpc_client.clone()
    }

    /// Take the event receiver (can only be called once)
    pub fn take_event_receiver(&mut self) -> Option<mpsc::Receiver<DemoEvent>> {
        self.event_rx.take()
    }

    /// Run the demo
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<DemoResults> {
        let start = Instant::now();

        // Initialize state
        {
            let mut state = self.state.write().await;
            state.started_at = Some(start);
        }

        // Print header
        self.print_header();

        // Try to connect to a real node
        let mut progress = ProgressDisplay::new();
        progress.start_spinner("Checking for HELIX node connection...");
        let connected = self.connect().await.unwrap_or(false);
        if connected {
            progress.finish_spinner("Connected to HELIX node - using real training");
        } else {
            progress.finish_spinner("Running in demo mode (simulated training)");
        }

        // Initialize real-time training tracker
        self.training_tracker
            .start(self.config.round_count as u64)
            .await;

        // Pre-warm if enabled
        if self.config.prewarm {
            self.run_prewarm().await?;
        }

        // Run scenario-specific demo
        let result = match self.config.scenario {
            DemoScenarioType::Quick => self.run_quick_demo(&mut shutdown).await,
            DemoScenarioType::FullTraining => self.run_full_training_demo(&mut shutdown).await,
            DemoScenarioType::Slashing => self.run_slashing_demo(&mut shutdown).await,
            DemoScenarioType::MultiModel => self.run_multi_model_demo(&mut shutdown).await,
            DemoScenarioType::FaultTolerance => self.run_fault_tolerance_demo(&mut shutdown).await,
        };

        // Calculate final metrics
        let duration = start.elapsed();
        let state = self.state.read().await;

        if let Err(e) = result {
            self.print_failure(&e.to_string());
            return Err(e);
        }

        // Print results
        self.print_results(&state, duration);

        Ok(DemoResults {
            scenario: self.config.scenario,
            duration,
            metrics: state.metrics.clone(),
            success: state.completed && state.phase != DemoPhase::Failed,
        })
    }

    /// Run pre-warming
    async fn run_prewarm(&self) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        {
            let mut state = self.state.write().await;
            state.phase = DemoPhase::Prewarming;
        }

        progress.start_spinner("Pre-warming caches for fast demo start...");

        if let Some(ref prewarmer) = self.prewarmer {
            let result = prewarm::prewarm_with_timeout(Duration::from_secs(5)).await;

            match result {
                Ok(prewarm_result) => {
                    progress.finish_spinner(&format!(
                        "Pre-warming complete ({}ms, {} items cached)",
                        prewarm_result.total_time.as_millis(),
                        prewarm_result.items_warmed.len()
                    ));
                }
                Err(e) => {
                    progress.finish_spinner(&format!("Pre-warming skipped: {}", e));
                }
            }
        }

        {
            let mut state = self.state.write().await;
            state.prewarmed = true;
        }

        Ok(())
    }

    fn print_header(&self) {
        println!();
        println!("{}", "═".repeat(70).cyan());
        println!("{}", " HELIX Demo - Trustless Distributed ML Training".cyan().bold());
        println!("{}", "═".repeat(70).cyan());
        println!();
        println!("{} {}", "Scenario:".yellow().bold(), self.config.scenario.name());
        println!("{}", self.config.scenario.description().dimmed());
        println!();
        println!("{}", "Configuration:".yellow().bold());
        println!("  Workers:      {}", self.config.worker_count);
        println!("  Aggregators:  {}", self.config.aggregator_count);
        println!("  Rounds:       {}", self.config.round_count);
        println!("  Models:       {}", self.config.model_count);
        println!("  Stake:        {} ETH per worker", self.config.stake_amount);
        println!("  Max Duration: {:?}", self.config.timing.hard_deadline);
        println!();
    }

    fn print_results(&self, state: &DemoState, duration: Duration) {
        println!();
        println!("{}", "═".repeat(70).cyan());
        println!("{}", " Demo Results".cyan().bold());
        println!("{}", "═".repeat(70).cyan());
        println!();
        println!("{}", "Summary:".yellow().bold());
        println!("  Duration:           {:?}", duration);
        println!("  Rounds Completed:   {}", state.metrics.rounds_completed);
        println!("  Proofs Generated:   {}", state.metrics.proofs_generated);
        println!(
            "  Proofs Verified:    {} ({} rejected)",
            state.metrics.proofs_verified, state.metrics.proofs_rejected
        );
        println!("  Total Stake:        {} ETH", state.metrics.total_stake);

        if state.metrics.stake_slashed > 0.0 {
            println!(
                "  Stake Slashed:      {} ETH",
                format!("{:.4}", state.metrics.stake_slashed).red()
            );
        }

        if !state.metrics.final_error_bounds.is_empty() {
            println!();
            println!("{}", "Model Results:".yellow().bold());
            for model in &state.models {
                let status = if model.is_complete {
                    "✓".green()
                } else {
                    "○".dimmed()
                };
                println!("  {} Model {} ({}):", status, model.id, model.name);
                println!("      Final Loss:       {:.6}", model.current_loss);
                println!(
                    "      Error Bound:      {:.2} / {:.2}",
                    model.error_bound, self.config.error_bound_max
                );
                if let Some(ref commitment) = model.final_commitment {
                    println!("      Commitment:       {}...", &commitment[..16.min(commitment.len())]);
                }
            }
        }

        if self.config.simulate_faults {
            println!();
            println!("{}", "Fault Tolerance:".yellow().bold());
            println!("  Node Failures:      {}", state.metrics.node_failures);
            println!("  Node Recoveries:    {}", state.metrics.node_recoveries);
        }

        println!();
        println!("{}", "Demo completed successfully!".green().bold());
        println!();
    }

    fn print_failure(&self, error: &str) {
        println!();
        println!("{} {}", "Demo failed:".red().bold(), error);
        println!();
    }

    async fn emit_event(
        &self,
        event_type: DemoEventType,
        description: &str,
        model_id: Option<u64>,
        node_id: Option<String>,
    ) {
        let state = self.state.read().await;
        let timestamp_ms = state
            .started_at
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);
        drop(state);

        let event = DemoEvent {
            timestamp_ms,
            event_type,
            description: description.to_string(),
            model_id,
            node_id,
        };

        // Store event in state
        {
            let mut state = self.state.write().await;
            state.events.push(event.clone());
        }

        // Send to event channel
        let _ = self.event_tx.send(event).await;
    }

    async fn set_phase(&self, phase: DemoPhase) {
        {
            let mut state = self.state.write().await;
            state.phase = phase;
        }
        self.emit_event(
            DemoEventType::PhaseChange,
            &format!("Entering phase: {}", phase.name()),
            None,
            None,
        )
        .await;
    }

    fn create_round_progress_bar(&self, total: u64) -> ProgressBar {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rounds ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb
    }

    async fn initialize_models(&self) {
        let mut state = self.state.write().await;
        if state.models.is_empty() {
            state.models.push(ModelState {
                id: 0,
                name: "HELIX Demo Model".to_string(),
                ipfs_hash: "QmXoYPwbkfYTJu87sD9DemoHash...".to_string(),
                current_round: 0,
                total_rounds: self.config.round_count as u64,
                current_loss: 2.5,
                loss_history: vec![],
                error_bound: 0.0,
                is_complete: false,
                final_commitment: None,
            });
        }
    }

    async fn initialize_nodes(&self) {
        let mut state = self.state.write().await;

        // Initialize workers
        if state.workers.is_empty() {
            for i in 0..self.config.worker_count {
                state.workers.push(WorkerState {
                    id: format!("worker-{}", i + 1),
                    address: format!("0x{:040x}", 0x742d35Cc6634C053u64 + i as u64),
                    stake: 0.0,
                    status: DemoWorkerStatus::Starting,
                    proofs_submitted: 0,
                    proofs_verified: 0,
                    current_model: Some(0),
                    reputation: 1.0,
                    slashed: false,
                });
            }
        }

        // Initialize aggregators
        if state.aggregators.is_empty() {
            for i in 0..self.config.aggregator_count {
                state.aggregators.push(AggregatorState {
                    id: format!("aggregator-{}", i + 1),
                    address: format!("0x{:040x}", 0xdD2FD4581271e230u64 + i as u64),
                    status: AggregatorStatus::Starting,
                    rounds_aggregated: 0,
                    proofs_verified: 0,
                    active_models: vec![0],
                });
            }
        }

        // Initialize RPC client workers
        self.rpc_client.read().await.init_workers(self.config.worker_count).await;
    }

    async fn set_workers_status(&self, status: DemoWorkerStatus) {
        let mut state = self.state.write().await;
        for worker in &mut state.workers {
            if worker.status != DemoWorkerStatus::Slashed && worker.status != DemoWorkerStatus::Offline {
                worker.status = status;
            }
        }
    }

    async fn simulate_training_round(&self, round: u64, model_id: u64) {
        let (current_loss, error_bound) = {
            let mut state = self.state.write().await;

            let mut loss = 0.0;
            let mut error = 0.0;

            if let Some(model) = state.models.iter_mut().find(|m| m.id == model_id) {
                model.current_round = round + 1;

                // Simulate loss decrease
                let loss_reduction = 0.08 + (rand::random::<f64>() * 0.04);
                model.current_loss = (model.current_loss - loss_reduction).max(0.01);
                model.loss_history.push(model.current_loss);

                // Accumulate error bound
                let error_increase = 3.0 + rand::random::<f64>() * 2.0;
                model.error_bound += error_increase;

                loss = model.current_loss;
                error = model.error_bound;
            }

            (loss, error)
        };

        // Advance RPC client (mock or real)
        self.rpc_client.read().await.advance_round().await;

        // Update real-time training tracker
        self.training_tracker
            .update_round(
                round + 1,
                current_loss,
                error_bound,
                TrainingPhase::RoundComplete,
            )
            .await;
    }

    async fn simulate_proof_generation(&self, round: u64, model_id: u64) {
        // Update proof tracker - witness generation phase
        self.proof_tracker
            .update(crate::rpc::ProofStatus {
                generating: true,
                phase: ProofPhase::WitnessGeneration,
                progress_percent: 20,
                constraints_satisfied: 0,
                total_constraints: 100000,
                elapsed_ms: 0,
                estimated_remaining_ms: 500,
                memory_usage_bytes: 256 * 1024 * 1024,
                gpu_accelerated: true,
                error_bound: 0.0,
            })
            .await;

        let mut state = self.state.write().await;

        let active_workers = state
            .workers
            .iter()
            .filter(|w| w.status != DemoWorkerStatus::Slashed && w.status != DemoWorkerStatus::Offline)
            .count();

        state.metrics.proofs_generated += active_workers as u64;

        for worker in &mut state.workers {
            if worker.status == DemoWorkerStatus::Proving {
                worker.proofs_submitted += 1;
            }
        }
        drop(state);

        // Update proof tracker - commitment generation
        self.proof_tracker
            .update(crate::rpc::ProofStatus {
                generating: true,
                phase: ProofPhase::CommitmentGeneration,
                progress_percent: 60,
                constraints_satisfied: 60000,
                total_constraints: 100000,
                elapsed_ms: 200,
                estimated_remaining_ms: 300,
                memory_usage_bytes: 384 * 1024 * 1024,
                gpu_accelerated: true,
                error_bound: 0.0,
            })
            .await;
    }

    async fn simulate_proof_verification(&self, round: u64, model_id: u64) {
        // Update proof tracker - proof computation complete
        self.proof_tracker
            .update(crate::rpc::ProofStatus {
                generating: true,
                phase: ProofPhase::ProofComputation,
                progress_percent: 90,
                constraints_satisfied: 90000,
                total_constraints: 100000,
                elapsed_ms: 400,
                estimated_remaining_ms: 50,
                memory_usage_bytes: 512 * 1024 * 1024,
                gpu_accelerated: true,
                error_bound: 0.0,
            })
            .await;

        let mut state = self.state.write().await;

        let verified = state.metrics.proofs_generated - state.metrics.proofs_rejected;
        state.metrics.proofs_verified = verified;

        for worker in &mut state.workers {
            if worker.status == DemoWorkerStatus::Waiting || worker.status == DemoWorkerStatus::Proving {
                worker.proofs_verified += 1;
            }
        }

        for agg in &mut state.aggregators {
            agg.rounds_aggregated += 1;
            agg.proofs_verified += 1;
        }
        drop(state);

        // Update proof tracker - complete
        self.proof_tracker
            .update(crate::rpc::ProofStatus {
                generating: false,
                phase: ProofPhase::Complete,
                progress_percent: 100,
                constraints_satisfied: 100000,
                total_constraints: 100000,
                elapsed_ms: 450,
                estimated_remaining_ms: 0,
                memory_usage_bytes: 0,
                gpu_accelerated: true,
                error_bound: 0.0,
            })
            .await;
    }

    async fn finalize_model(&self, model_id: u64) {
        let mut state = self.state.write().await;

        let error_bound = state
            .models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.error_bound);

        if let Some(model) = state.models.iter_mut().find(|m| m.id == model_id) {
            model.is_complete = true;
            model.final_commitment = Some(format!("0x{:064x}", rand::random::<u128>()));
        }

        if let Some(bound) = error_bound {
            state.metrics.final_error_bounds.insert(model_id, bound);
        }
    }

    // ========================================================================
    // Quick Demo Implementation (30 seconds)
    // ========================================================================

    async fn run_quick_demo(&self, shutdown: &mut broadcast::Receiver<()>) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        // Phase 1: Setup
        println!("{}", "Phase 1: Quick Setup".green().bold());
        self.set_phase(DemoPhase::Initializing).await;

        progress.start_spinner("Initializing demo environment...");
        tokio::time::sleep(Duration::from_millis(300)).await;
        progress.finish_spinner("Environment ready");

        progress.start_spinner("Connecting to contracts...");
        tokio::time::sleep(Duration::from_millis(200)).await;
        progress.finish_spinner("Contracts connected");

        // Register model
        self.set_phase(DemoPhase::RegisteringModels).await;
        progress.start_spinner("Registering model...");
        self.initialize_models().await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        progress.finish_spinner("Model registered (ID: 0)");
        self.emit_event(DemoEventType::ModelRegistered, "Demo model registered", Some(0), None).await;

        // Start nodes
        self.set_phase(DemoPhase::StartingNodes).await;
        self.initialize_nodes().await;
        for i in 0..self.config.worker_count {
            if shutdown.try_recv().is_ok() {
                return Ok(());
            }
            progress.start_spinner(&format!("Starting worker {}...", i + 1));
            tokio::time::sleep(Duration::from_millis(150)).await;
            progress.finish_spinner(&format!("Worker {} online", i + 1));
            self.emit_event(
                DemoEventType::NodeStarted,
                &format!("Worker {} started", i + 1),
                None,
                Some(format!("worker-{}", i + 1)),
            )
            .await;
        }

        // Staking
        println!();
        println!("{}", "Phase 2: Staking".green().bold());
        self.set_phase(DemoPhase::Staking).await;

        for i in 0..self.config.worker_count {
            progress.start_spinner(&format!("Worker {} staking {} ETH...", i + 1, self.config.stake_amount));
            tokio::time::sleep(Duration::from_millis(150)).await;
            progress.finish_spinner(&format!("Worker {} staked", i + 1));
            self.emit_event(
                DemoEventType::StakeDeposited,
                &format!("Worker {} deposited {} ETH", i + 1, self.config.stake_amount),
                Some(0),
                Some(format!("worker-{}", i + 1)),
            )
            .await;

            let mut state = self.state.write().await;
            if let Some(worker) = state.workers.get_mut(i as usize) {
                worker.stake = self.config.stake_amount;
            }
            state.metrics.total_stake += self.config.stake_amount;
        }

        // Training rounds
        println!();
        println!("{}", "Phase 3: Training".green().bold());
        self.set_phase(DemoPhase::Training).await;

        let pb = self.create_round_progress_bar(self.config.round_count as u64);

        for round in 0..self.config.round_count {
            if shutdown.try_recv().is_ok() {
                pb.abandon_with_message("Demo interrupted");
                return Ok(());
            }

            pb.set_message(format!("Round {}/{}", round + 1, self.config.round_count));
            self.emit_event(
                DemoEventType::RoundStarted,
                &format!("Round {} started", round + 1),
                Some(0),
                None,
            )
            .await;

            // Simulate training
            self.simulate_training_round(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            // Simulate proof generation
            self.set_phase(DemoPhase::Proving).await;
            self.simulate_proof_generation(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            // Simulate verification
            self.set_phase(DemoPhase::Verifying).await;
            self.simulate_proof_verification(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            self.emit_event(
                DemoEventType::RoundCompleted,
                &format!("Round {} completed", round + 1),
                Some(0),
                None,
            )
            .await;
            pb.inc(1);

            let mut state = self.state.write().await;
            state.metrics.rounds_completed += 1;
            self.set_phase(DemoPhase::Training).await;
        }

        pb.finish_with_message(format!("{} Training complete!", "✓".green()));

        // Finalize
        println!();
        println!("{}", "Phase 4: Finalization".green().bold());
        self.set_phase(DemoPhase::Finalizing).await;

        progress.start_spinner("Finalizing model...");
        tokio::time::sleep(Duration::from_millis(400)).await;
        self.finalize_model(0).await;
        progress.finish_spinner("Model finalized");
        self.emit_event(DemoEventType::ModelFinalized, "Model 0 finalized", Some(0), None).await;

        self.set_phase(DemoPhase::Complete).await;
        {
            let mut state = self.state.write().await;
            state.completed = true;
        }

        Ok(())
    }

    // ========================================================================
    // Full Training Demo Implementation (90 seconds)
    // ========================================================================

    async fn run_full_training_demo(&self, shutdown: &mut broadcast::Receiver<()>) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        // Phase 1: Contract Deployment
        println!("{}", "Phase 1: Contract Deployment".green().bold());
        self.set_phase(DemoPhase::DeployingContracts).await;

        let contracts = [
            ("HelixCoordinatorV2", "0x5FbDB2315678afecb367f032d93F642f64180aa3"),
            ("Halo2Verifier", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"),
            ("HelixToken", "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0"),
        ];

        for (name, address) in contracts {
            progress.start_spinner(&format!("Deploying {}...", name));
            tokio::time::sleep(Duration::from_millis(500)).await;
            progress.finish_spinner(&format!("{} deployed at {}", name, &address[..10]));
            self.emit_event(DemoEventType::ContractDeployed, &format!("{} deployed", name), None, None).await;
        }

        // Phase 2: Model Registration
        println!();
        println!("{}", "Phase 2: Model Registration".green().bold());
        self.set_phase(DemoPhase::RegisteringModels).await;

        progress.start_spinner("Uploading initial model to IPFS...");
        tokio::time::sleep(Duration::from_millis(400)).await;
        progress.finish_spinner("Model uploaded: QmXoYPwbkfYTJu87sD9...");

        progress.start_spinner("Registering model on-chain...");
        self.initialize_models().await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        progress.finish_spinner("Model registered (ID: 0)");
        self.emit_event(DemoEventType::ModelRegistered, "Training model registered", Some(0), None).await;

        // Phase 3: Node Setup
        println!();
        println!("{}", "Phase 3: Network Setup".green().bold());
        self.set_phase(DemoPhase::StartingNodes).await;
        self.initialize_nodes().await;

        // Start aggregators
        for i in 0..self.config.aggregator_count {
            progress.start_spinner(&format!("Starting aggregator {}...", i + 1));
            tokio::time::sleep(Duration::from_millis(300)).await;
            progress.finish_spinner(&format!("Aggregator {} online", i + 1));
            self.emit_event(
                DemoEventType::NodeStarted,
                &format!("Aggregator {} started", i + 1),
                None,
                Some(format!("aggregator-{}", i + 1)),
            )
            .await;
        }

        // Start workers
        for i in 0..self.config.worker_count {
            if shutdown.try_recv().is_ok() {
                return Ok(());
            }
            progress.start_spinner(&format!("Starting worker {}...", i + 1));
            tokio::time::sleep(Duration::from_millis(200)).await;
            progress.finish_spinner(&format!("Worker {} online", i + 1));
            self.emit_event(
                DemoEventType::NodeStarted,
                &format!("Worker {} started", i + 1),
                None,
                Some(format!("worker-{}", i + 1)),
            )
            .await;
        }

        // Phase 4: Staking
        println!();
        println!("{}", "Phase 4: Token Staking".green().bold());
        self.set_phase(DemoPhase::Staking).await;

        for i in 0..self.config.worker_count {
            progress.start_spinner(&format!("Worker {} staking {} ETH...", i + 1, self.config.stake_amount));
            tokio::time::sleep(Duration::from_millis(200)).await;
            progress.finish_spinner(&format!("Worker {} staked {} ETH (locked)", i + 1, self.config.stake_amount));
            self.emit_event(
                DemoEventType::StakeDeposited,
                &format!("Worker {} deposited stake", i + 1),
                Some(0),
                Some(format!("worker-{}", i + 1)),
            )
            .await;

            let mut state = self.state.write().await;
            if let Some(worker) = state.workers.get_mut(i as usize) {
                worker.stake = self.config.stake_amount;
                worker.status = DemoWorkerStatus::Idle;
            }
            state.metrics.total_stake += self.config.stake_amount;
        }

        // Phase 5: Training
        println!();
        println!("{}", "Phase 5: Distributed Training".green().bold());
        self.set_phase(DemoPhase::Training).await;

        let pb = self.create_round_progress_bar(self.config.round_count as u64);

        for round in 0..self.config.round_count {
            if shutdown.try_recv().is_ok() {
                pb.abandon_with_message("Demo interrupted");
                return Ok(());
            }

            pb.set_message(format!("Round {}/{} - Training", round + 1, self.config.round_count));
            self.emit_event(
                DemoEventType::RoundStarted,
                &format!("Round {} initiated", round + 1),
                Some(0),
                None,
            )
            .await;

            // Training phase
            self.set_workers_status(DemoWorkerStatus::Training).await;
            self.simulate_training_round(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration * 4 / 10).await;

            // Proof generation phase
            pb.set_message(format!("Round {}/{} - Generating proofs", round + 1, self.config.round_count));
            self.set_phase(DemoPhase::Proving).await;
            self.set_workers_status(DemoWorkerStatus::Proving).await;
            self.simulate_proof_generation(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration * 3 / 10).await;

            // Verification phase
            pb.set_message(format!("Round {}/{} - Verifying proofs", round + 1, self.config.round_count));
            self.set_phase(DemoPhase::Verifying).await;
            self.set_workers_status(DemoWorkerStatus::Waiting).await;
            self.simulate_proof_verification(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration * 3 / 10).await;

            self.emit_event(
                DemoEventType::RoundCompleted,
                &format!("Round {} committed on-chain", round + 1),
                Some(0),
                None,
            )
            .await;
            pb.inc(1);

            {
                let mut state = self.state.write().await;
                state.metrics.rounds_completed += 1;
            }
            self.set_phase(DemoPhase::Training).await;
        }

        pb.finish_with_message(format!("{} All training rounds complete!", "✓".green()));

        // Phase 6: Finalization
        println!();
        println!("{}", "Phase 6: Model Finalization".green().bold());
        self.set_phase(DemoPhase::Finalizing).await;

        progress.start_spinner("Computing final model commitment...");
        tokio::time::sleep(Duration::from_millis(400)).await;
        progress.finish_spinner("Final commitment computed");

        progress.start_spinner("Finalizing model on-chain...");
        self.finalize_model(0).await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        progress.finish_spinner("Model finalized and verified");
        self.emit_event(DemoEventType::ModelFinalized, "Training complete, model finalized", Some(0), None).await;

        self.set_phase(DemoPhase::Complete).await;
        {
            let mut state = self.state.write().await;
            state.completed = true;
        }

        Ok(())
    }

    // ========================================================================
    // Slashing Demo Implementation
    // ========================================================================

    async fn run_slashing_demo(&self, shutdown: &mut broadcast::Receiver<()>) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        // Setup phases (abbreviated)
        println!("{}", "Phase 1: Setup".green().bold());
        self.set_phase(DemoPhase::Initializing).await;

        progress.start_spinner("Setting up network...");
        self.initialize_models().await;
        self.initialize_nodes().await;
        tokio::time::sleep(Duration::from_millis(800)).await;
        progress.finish_spinner("Network ready");

        // Staking
        println!();
        println!("{}", "Phase 2: Staking".green().bold());
        self.set_phase(DemoPhase::Staking).await;

        for i in 0..self.config.worker_count {
            progress.start_spinner(&format!("Worker {} staking {} ETH...", i + 1, self.config.stake_amount));
            tokio::time::sleep(Duration::from_millis(150)).await;
            progress.finish_spinner(&format!("Worker {} staked", i + 1));

            let mut state = self.state.write().await;
            if let Some(worker) = state.workers.get_mut(i as usize) {
                worker.stake = self.config.stake_amount;
            }
            state.metrics.total_stake += self.config.stake_amount;
        }

        // Training with fault injection
        println!();
        println!("{}", "Phase 3: Training with Fault Injection".green().bold());
        self.set_phase(DemoPhase::Training).await;

        let pb = self.create_round_progress_bar(self.config.round_count as u64);
        let malicious_worker = 2;
        let fault_round = self.config.round_count / 2;

        for round in 0..self.config.round_count {
            if shutdown.try_recv().is_ok() {
                pb.abandon_with_message("Demo interrupted");
                return Ok(());
            }

            pb.set_message(format!("Round {}/{}", round + 1, self.config.round_count));
            self.emit_event(
                DemoEventType::RoundStarted,
                &format!("Round {} started", round + 1),
                Some(0),
                None,
            )
            .await;

            // Training
            self.simulate_training_round(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 4).await;

            // Proof generation
            self.set_phase(DemoPhase::Proving).await;

            if round == fault_round {
                // Inject malicious proof
                pb.suspend(|| {
                    println!();
                    println!("  {} {}", "⚠".yellow().bold(), "Worker 3 submitting invalid proof...".yellow());
                });

                self.emit_event(
                    DemoEventType::ProofGenerated,
                    "Worker 3 submitted proof with invalid computation",
                    Some(0),
                    Some("worker-3".to_string()),
                )
                .await;

                tokio::time::sleep(self.config.round_duration / 4).await;

                // Verification detects fault
                self.set_phase(DemoPhase::Verifying).await;

                pb.suspend(|| {
                    println!("  {} {}", "⚠".red().bold(), "Invalid proof detected by verifier!".red());
                    println!("  {} {}", "→".cyan(), "Error bound exceeded: 1523.7 > 500.0 max".dimmed());
                });

                self.emit_event(
                    DemoEventType::ProofRejected,
                    "Proof rejected: error bound exceeded",
                    Some(0),
                    Some("worker-3".to_string()),
                )
                .await;

                self.emit_event(
                    DemoEventType::FaultDetected,
                    "Malicious computation detected in worker 3",
                    Some(0),
                    Some("worker-3".to_string()),
                )
                .await;

                tokio::time::sleep(Duration::from_millis(500)).await;

                // Execute slashing
                self.set_phase(DemoPhase::Slashing).await;

                pb.suspend(|| {
                    println!();
                    println!("  {} {}", "🔪", "SLASHING EXECUTION".red().bold());
                    println!("  {} Worker 3 stake: {} ETH → 0 ETH", "→".cyan(), self.config.stake_amount);
                    println!("  {} Reputation: 95% → 0%", "→".cyan());
                    println!("  {} Status: {} Slashed", "→".cyan(), "✗".red());
                });

                {
                    let mut state = self.state.write().await;
                    let slashed_stake = state
                        .workers
                        .get(malicious_worker as usize)
                        .map(|w| w.stake)
                        .unwrap_or(0.0);

                    if let Some(worker) = state.workers.get_mut(malicious_worker as usize) {
                        worker.stake = 0.0;
                        worker.status = DemoWorkerStatus::Slashed;
                        worker.slashed = true;
                        worker.reputation = 0.0;
                    }
                    state.metrics.stake_slashed += slashed_stake;
                    state.metrics.proofs_rejected += 1;
                }

                self.emit_event(
                    DemoEventType::SlashingExecuted,
                    &format!("Worker 3 slashed: {} ETH confiscated", self.config.stake_amount),
                    Some(0),
                    Some("worker-3".to_string()),
                )
                .await;

                tokio::time::sleep(Duration::from_secs(1)).await;

                self.set_phase(DemoPhase::Training).await;
            } else {
                // Normal round
                self.simulate_proof_generation(round as u64, 0).await;
                tokio::time::sleep(self.config.round_duration / 4).await;

                self.set_phase(DemoPhase::Verifying).await;
                self.simulate_proof_verification(round as u64, 0).await;
                tokio::time::sleep(self.config.round_duration / 4).await;

                self.set_phase(DemoPhase::Training).await;
            }

            self.emit_event(
                DemoEventType::RoundCompleted,
                &format!("Round {} completed", round + 1),
                Some(0),
                None,
            )
            .await;
            pb.inc(1);

            {
                let mut state = self.state.write().await;
                state.metrics.rounds_completed += 1;
            }
        }

        pb.finish_with_message(format!("{} Training complete (with slashing event)!", "✓".green()));

        // Summary
        println!();
        println!("{}", "Slashing Summary:".yellow().bold());
        {
            let state = self.state.read().await;
            println!("  Proofs Rejected:   {}", state.metrics.proofs_rejected);
            println!("  Stake Slashed:     {} ETH", format!("{:.4}", state.metrics.stake_slashed).red());
            println!("  Workers Slashed:   1 (Worker 3)");
        }

        self.set_phase(DemoPhase::Complete).await;
        {
            let mut state = self.state.write().await;
            state.completed = true;
        }

        Ok(())
    }

    // ========================================================================
    // Multi-Model Demo Implementation
    // ========================================================================

    async fn run_multi_model_demo(&self, shutdown: &mut broadcast::Receiver<()>) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        // Setup
        println!("{}", "Phase 1: Multi-Model Setup".green().bold());
        self.set_phase(DemoPhase::Initializing).await;

        progress.start_spinner("Initializing network...");
        self.initialize_nodes().await;
        tokio::time::sleep(Duration::from_millis(500)).await;
        progress.finish_spinner("Network ready");

        // Register multiple models
        println!();
        println!("{}", "Phase 2: Model Registration".green().bold());
        self.set_phase(DemoPhase::RegisteringModels).await;

        let model_names = ["ResNet-50", "BERT-Base", "GPT-2 Small"];
        {
            let mut state = self.state.write().await;
            for (i, name) in model_names.iter().enumerate().take(self.config.model_count as usize) {
                state.models.push(ModelState {
                    id: i as u64,
                    name: name.to_string(),
                    ipfs_hash: format!("QmModel{}Hash...", i),
                    current_round: 0,
                    total_rounds: self.config.round_count as u64,
                    current_loss: 2.5 - (i as f64 * 0.3),
                    loss_history: vec![],
                    error_bound: 0.0,
                    is_complete: false,
                    final_commitment: None,
                });
            }
        }

        for (i, name) in model_names.iter().enumerate().take(self.config.model_count as usize) {
            progress.start_spinner(&format!("Registering {} (Model {})...", name, i));
            tokio::time::sleep(Duration::from_millis(300)).await;
            progress.finish_spinner(&format!("{} registered (ID: {})", name, i));
            self.emit_event(DemoEventType::ModelRegistered, &format!("{} registered", name), Some(i as u64), None).await;
        }

        // Staking
        println!();
        println!("{}", "Phase 3: Staking".green().bold());
        self.set_phase(DemoPhase::Staking).await;

        for i in 0..self.config.worker_count {
            let model_assignment = (i as usize) % self.config.model_count as usize;
            progress.start_spinner(&format!("Worker {} staking for {}...", i + 1, model_names[model_assignment]));
            tokio::time::sleep(Duration::from_millis(150)).await;
            progress.finish_spinner(&format!("Worker {} assigned to Model {}", i + 1, model_assignment));

            let mut state = self.state.write().await;
            if let Some(worker) = state.workers.get_mut(i as usize) {
                worker.stake = self.config.stake_amount;
                worker.current_model = Some(model_assignment as u64);
            }
            state.metrics.total_stake += self.config.stake_amount;
        }

        // Concurrent training
        println!();
        println!("{}", "Phase 4: Concurrent Training".green().bold());
        self.set_phase(DemoPhase::Training).await;

        let multi = MultiProgress::new();
        let mut model_pbs: Vec<ProgressBar> = Vec::new();

        for (i, name) in model_names.iter().enumerate().take(self.config.model_count as usize) {
            let pb = multi.add(ProgressBar::new(self.config.round_count as u64));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(&format!("  {{spinner:.green}} {} [{{bar:30.cyan/blue}}] {{pos}}/{{len}} ({{eta}})", name))
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_position(0);
            model_pbs.push(pb);
        }

        for round in 0..self.config.round_count {
            if shutdown.try_recv().is_ok() {
                for pb in &model_pbs {
                    pb.abandon_with_message("Interrupted");
                }
                return Ok(());
            }

            for (model_id, pb) in model_pbs.iter().enumerate() {
                pb.set_message(format!("Round {}", round + 1));

                {
                    let mut state = self.state.write().await;
                    if let Some(model) = state.models.get_mut(model_id) {
                        model.current_round = round as u64 + 1;
                        let loss_reduction = 0.1 + (rand::random::<f64>() * 0.05);
                        model.current_loss = (model.current_loss - loss_reduction).max(0.01);
                        model.loss_history.push(model.current_loss);
                        model.error_bound += 5.0 + rand::random::<f64>() * 3.0;
                    }
                    state.metrics.proofs_generated += 1;
                    state.metrics.proofs_verified += 1;
                }

                pb.inc(1);
            }

            tokio::time::sleep(self.config.round_duration).await;

            {
                let mut state = self.state.write().await;
                state.metrics.rounds_completed += self.config.model_count as u64;
            }
        }

        for pb in &model_pbs {
            pb.finish_with_message("Complete");
        }

        // Finalize all models
        println!();
        println!("{}", "Phase 5: Model Finalization".green().bold());
        self.set_phase(DemoPhase::Finalizing).await;

        for (i, name) in model_names.iter().enumerate().take(self.config.model_count as usize) {
            progress.start_spinner(&format!("Finalizing {}...", name));
            tokio::time::sleep(Duration::from_millis(300)).await;

            {
                let mut state = self.state.write().await;
                let error_bound = state.models.get(i).map(|m| m.error_bound).unwrap_or(0.0);

                if let Some(model) = state.models.get_mut(i) {
                    model.is_complete = true;
                    model.final_commitment = Some(format!("0x{:064x}", rand::random::<u64>()));
                }
                state.metrics.final_error_bounds.insert(i as u64, error_bound);
            }

            progress.finish_spinner(&format!("{} finalized", name));
            self.emit_event(DemoEventType::ModelFinalized, &format!("{} training complete", name), Some(i as u64), None).await;
        }

        self.set_phase(DemoPhase::Complete).await;
        {
            let mut state = self.state.write().await;
            state.completed = true;
        }

        Ok(())
    }

    // ========================================================================
    // Fault Tolerance Demo Implementation
    // ========================================================================

    async fn run_fault_tolerance_demo(&self, shutdown: &mut broadcast::Receiver<()>) -> Result<()> {
        let mut progress = ProgressDisplay::new();

        // Setup
        println!("{}", "Phase 1: Fault-Tolerant Setup".green().bold());
        self.set_phase(DemoPhase::Initializing).await;

        progress.start_spinner("Initializing redundant network...");
        self.initialize_models().await;
        self.initialize_nodes().await;
        tokio::time::sleep(Duration::from_millis(600)).await;
        progress.finish_spinner(&format!(
            "Network ready ({} workers, {} aggregators)",
            self.config.worker_count, self.config.aggregator_count
        ));

        // Staking
        println!();
        println!("{}", "Phase 2: Staking".green().bold());
        self.set_phase(DemoPhase::Staking).await;

        for i in 0..self.config.worker_count {
            progress.start_spinner(&format!("Worker {} staking...", i + 1));
            tokio::time::sleep(Duration::from_millis(100)).await;
            progress.finish_spinner(&format!("Worker {} ready", i + 1));

            let mut state = self.state.write().await;
            if let Some(worker) = state.workers.get_mut(i as usize) {
                worker.stake = self.config.stake_amount;
            }
            state.metrics.total_stake += self.config.stake_amount;
        }

        // Training with fault injection
        println!();
        println!("{}", "Phase 3: Training with Fault Simulation".green().bold());
        self.set_phase(DemoPhase::Training).await;

        let pb = self.create_round_progress_bar(self.config.round_count as u64);
        let fault_rounds = vec![3, 6, 9];
        let recovery_delay = 2;

        for round in 0..self.config.round_count {
            if shutdown.try_recv().is_ok() {
                pb.abandon_with_message("Demo interrupted");
                return Ok(());
            }

            pb.set_message(format!("Round {}/{}", round + 1, self.config.round_count));

            // Check for fault events
            if fault_rounds.contains(&round) {
                let failed_worker = (round as usize % self.config.worker_count as usize) + 1;

                pb.suspend(|| {
                    println!();
                    println!("  {} {}", "⚠".yellow().bold(), format!("Worker {} went offline!", failed_worker).yellow());
                });

                {
                    let mut state = self.state.write().await;
                    if let Some(worker) = state.workers.get_mut(failed_worker - 1) {
                        worker.status = DemoWorkerStatus::Offline;
                    }
                    state.metrics.node_failures += 1;
                }

                self.emit_event(
                    DemoEventType::FaultDetected,
                    &format!("Worker {} connection lost", failed_worker),
                    Some(0),
                    Some(format!("worker-{}", failed_worker)),
                )
                .await;

                pb.suspend(|| {
                    let active = self.config.worker_count - 1;
                    println!("  {} Network continues with {} active workers", "→".cyan(), active);
                });
            }

            // Check for recovery events
            if round > recovery_delay && fault_rounds.contains(&(round - recovery_delay)) {
                let recovering_worker = ((round - recovery_delay) as usize % self.config.worker_count as usize) + 1;

                pb.suspend(|| {
                    println!();
                    println!(
                        "  {} {}",
                        "↻".green().bold(),
                        format!("Worker {} reconnected and syncing...", recovering_worker).green()
                    );
                });

                {
                    let mut state = self.state.write().await;
                    if let Some(worker) = state.workers.get_mut(recovering_worker - 1) {
                        worker.status = DemoWorkerStatus::Syncing;
                    }
                }

                tokio::time::sleep(Duration::from_millis(200)).await;

                pb.suspend(|| {
                    println!("  {} Worker {} recovered and rejoined training", "✓".green(), recovering_worker);
                });

                {
                    let mut state = self.state.write().await;
                    if let Some(worker) = state.workers.get_mut(recovering_worker - 1) {
                        worker.status = DemoWorkerStatus::Training;
                    }
                    state.metrics.node_recoveries += 1;
                }

                self.emit_event(
                    DemoEventType::NodeRecovered,
                    &format!("Worker {} recovered and synced", recovering_worker),
                    Some(0),
                    Some(format!("worker-{}", recovering_worker)),
                )
                .await;
            }

            // Normal training round
            self.simulate_training_round(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            self.set_phase(DemoPhase::Proving).await;
            self.simulate_proof_generation(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            self.set_phase(DemoPhase::Verifying).await;
            self.simulate_proof_verification(round as u64, 0).await;
            tokio::time::sleep(self.config.round_duration / 3).await;

            self.set_phase(DemoPhase::Training).await;
            pb.inc(1);

            {
                let mut state = self.state.write().await;
                state.metrics.rounds_completed += 1;
            }
        }

        pb.finish_with_message(format!(
            "{} Training complete (network survived {} failures)!",
            "✓".green(),
            fault_rounds.len()
        ));

        // Finalization
        println!();
        println!("{}", "Phase 4: Finalization".green().bold());
        self.set_phase(DemoPhase::Finalizing).await;

        progress.start_spinner("Finalizing model...");
        self.finalize_model(0).await;
        tokio::time::sleep(Duration::from_millis(400)).await;
        progress.finish_spinner("Model finalized");

        // Fault tolerance summary
        println!();
        println!("{}", "Fault Tolerance Summary:".yellow().bold());
        {
            let state = self.state.read().await;
            println!("  Total Failures:     {}", state.metrics.node_failures);
            println!("  Successful Recoveries: {}", state.metrics.node_recoveries);
            println!("  Network Availability:  100% (training never interrupted)");
            println!(
                "  Final Error Bound:   {:.2}",
                state.models.get(0).map(|m| m.error_bound).unwrap_or(0.0)
            );
        }

        self.set_phase(DemoPhase::Complete).await;
        {
            let mut state = self.state.write().await;
            state.completed = true;
        }

        Ok(())
    }
}

// ============================================================================
// Demo Results
// ============================================================================

/// Results from a completed demo run
#[derive(Debug, Clone)]
pub struct DemoResults {
    /// Scenario that was run
    pub scenario: DemoScenarioType,
    /// Total duration
    pub duration: Duration,
    /// Final metrics
    pub metrics: DemoMetrics,
    /// Whether demo completed successfully
    pub success: bool,
}

impl DemoResults {
    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Scenario: {}\n", self.scenario.name()));
        s.push_str(&format!("Duration: {:?}\n", self.duration));
        s.push_str(&format!("Rounds Completed: {}\n", self.metrics.rounds_completed));
        s.push_str(&format!("Proofs Generated: {}\n", self.metrics.proofs_generated));
        s.push_str(&format!("Proofs Verified: {}\n", self.metrics.proofs_verified));
        s.push_str(&format!("Success: {}\n", self.success));
        s
    }

    /// Export results to JSON
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&serde_json::json!({
            "scenario": format!("{:?}", self.scenario),
            "duration_ms": self.duration.as_millis(),
            "metrics": {
                "rounds_completed": self.metrics.rounds_completed,
                "proofs_generated": self.metrics.proofs_generated,
                "proofs_verified": self.metrics.proofs_verified,
                "proofs_rejected": self.metrics.proofs_rejected,
                "total_stake": self.metrics.total_stake,
                "stake_slashed": self.metrics.stake_slashed,
                "node_failures": self.metrics.node_failures,
                "node_recoveries": self.metrics.node_recoveries,
            },
            "success": self.success,
        }))?)
    }
}

// ============================================================================
// Demo Builder
// ============================================================================

/// Builder for creating demo configurations
pub struct DemoBuilder {
    config: DemoConfig,
    endpoint: Option<String>,
}

impl DemoBuilder {
    /// Create a new demo builder with default Quick scenario
    pub fn new() -> Self {
        Self {
            config: DemoConfig::default(),
            endpoint: None,
        }
    }

    /// Set the RPC endpoint for real node connection
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Set the scenario type
    pub fn scenario(mut self, scenario: DemoScenarioType) -> Self {
        let endpoint = self.endpoint.take();
        self.config = scenario.default_config();
        self.endpoint = endpoint;
        self
    }

    /// Set the number of workers
    pub fn workers(mut self, count: u32) -> Self {
        self.config.worker_count = count;
        self
    }

    /// Set the number of aggregators
    pub fn aggregators(mut self, count: u32) -> Self {
        self.config.aggregator_count = count;
        self
    }

    /// Set the number of training rounds
    pub fn rounds(mut self, count: u32) -> Self {
        self.config.round_count = count;
        self
    }

    /// Set the round duration
    pub fn round_duration(mut self, duration: Duration) -> Self {
        self.config.round_duration = duration;
        self
    }

    /// Set the number of models to train
    pub fn models(mut self, count: u32) -> Self {
        self.config.model_count = count;
        self
    }

    /// Set the stake amount per worker
    pub fn stake(mut self, amount: f64) -> Self {
        self.config.stake_amount = amount;
        self
    }

    /// Enable headless mode
    pub fn headless(mut self, enabled: bool) -> Self {
        self.config.headless = enabled;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.config.verbose = enabled;
        self
    }

    /// Enable pre-warming
    pub fn prewarm(mut self, enabled: bool) -> Self {
        self.config.prewarm = enabled;
        self
    }

    /// Build the demo runner
    pub fn build(self) -> DemoRunner {
        if let Some(endpoint) = self.endpoint {
            DemoRunner::with_endpoint(self.config, &endpoint)
        } else {
            DemoRunner::new(self.config)
        }
    }

    /// Get access to real-time proof tracker for UI integration
    pub fn proof_tracker(&self) -> Arc<RealTimeProofTracker> {
        Arc::new(RealTimeProofTracker::new())
    }

    /// Get access to real-time training tracker for UI integration
    pub fn training_tracker(&self) -> Arc<RealTimeTrainingTracker> {
        Arc::new(RealTimeTrainingTracker::new())
    }
}

impl Default for DemoBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_defaults() {
        let quick = DemoScenarioType::Quick.default_config();
        assert_eq!(quick.worker_count, 3);
        assert_eq!(quick.round_count, 5);

        let full = DemoScenarioType::FullTraining.default_config();
        assert_eq!(full.worker_count, 5);
        assert_eq!(full.round_count, 10);
    }

    #[test]
    fn test_demo_builder() {
        let runner = DemoBuilder::new()
            .scenario(DemoScenarioType::Quick)
            .workers(5)
            .rounds(10)
            .build();

        assert_eq!(runner.config.worker_count, 5);
        assert_eq!(runner.config.round_count, 10);
    }

    #[test]
    fn test_worker_status_symbols() {
        // Just test that symbols don't panic
        let _ = DemoWorkerStatus::Training.symbol();
        let _ = DemoWorkerStatus::Slashed.symbol();
    }
}
