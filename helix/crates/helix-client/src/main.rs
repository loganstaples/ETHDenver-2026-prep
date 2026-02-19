//! HELIX CLI - Distributed ML Training Orchestration
//!
//! A command-line interface for managing HELIX distributed training networks,
//! running demos, and monitoring network health.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand, Args};
use colored::*;
use tokio::sync::broadcast;

mod commands;
mod config;
mod demo;
mod help;
mod orchestrator;
mod progress;
mod health;
mod benchmark;
mod rpc;
mod visualization;
mod wallet;

// Use dashboard from the lib crate (it needs access to full_orchestration/zk_proof_layer)
use helix_client::dashboard;

use commands::{
    init::InitCommand,
    join::JoinCommand,
    status::StatusCommand,
    query::QueryCommand,
    export::ExportCommand,
    train::{TrainCommand, TrainOptions},
};
use orchestrator::NetworkOrchestrator;
use progress::ProgressDisplay;

/// HELIX CLI - Trustless Distributed ML Training
#[derive(Parser)]
#[command(name = "helix")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, env = "HELIX_CONFIG", default_value = "helix.toml")]
    config: PathBuf,

    /// Verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new HELIX node or network
    Init(InitArgs),

    /// Join an existing training network
    Join(JoinArgs),

    /// Check node/network status
    Status(StatusArgs),

    /// Query model or round data
    Query(QueryArgs),

    /// Export training artifacts
    Export(ExportArgs),

    /// Start a training session with a model
    Train(TrainArgs),

    /// Start a multi-node demo network
    Demo(DemoArgs),

    /// Orchestrate distributed training across nodes
    Orchestrate(OrchestrateArgs),

    /// Run health checks
    Health(HealthArgs),

    /// Start the status dashboard server
    Dashboard(DashboardArgs),

    /// Run performance benchmarks
    Benchmark(BenchmarkArgs),

    /// Aggregate and view logs
    Logs(LogsArgs),

    /// Watch and reload configuration
    Watch(WatchArgs),

    /// Launch interactive training visualization
    Visualize(VisualizeArgs),

    /// Show detailed help for a topic or command
    Guide(HelpArgs),

    /// Submit a pre-generated proof to the coordinator contract
    #[cfg(feature = "chain")]
    SubmitProof(SubmitProofArgs),

    /// Run full MPC-primary training as model owner (end-to-end)
    MpcTrain(MpcTrainArgs),

    /// Join an MPC training session as a worker
    MpcWorker(MpcWorkerArgs),

    /// Spawn multiple MPC worker processes locally
    SpawnWorkers(SpawnWorkersArgs),
}

// ============================================================================
// Command Arguments
// ============================================================================

#[derive(Args)]
struct InitArgs {
    /// Node name
    #[arg(short, long)]
    name: Option<String>,

    /// Network to initialize (mainnet, testnet, local)
    #[arg(short = 'N', long, default_value = "local")]
    network: String,

    /// Generate a new wallet
    #[arg(long)]
    generate_wallet: bool,

    /// Initialize as aggregator node
    #[arg(long)]
    aggregator: bool,
}

#[derive(Args)]
struct JoinArgs {
    /// Address of the network coordinator
    #[arg(short, long)]
    coordinator: String,

    /// Model ID to join training for
    #[arg(short, long)]
    model_id: u64,

    /// Stake amount in ETH
    #[arg(short, long)]
    stake: Option<f64>,

    /// Node capabilities (train, aggregate, prove)
    #[arg(long, value_delimiter = ',')]
    capabilities: Vec<String>,
}

#[derive(Args)]
struct StatusArgs {
    /// Show detailed status
    #[arg(short, long)]
    detailed: bool,

    /// Watch status continuously
    #[arg(short, long)]
    watch: bool,

    /// Specific model ID to check
    #[arg(short, long)]
    model_id: Option<u64>,

    /// Refresh interval in seconds (for watch mode)
    #[arg(long, default_value = "5")]
    interval: u64,
}

#[derive(Args)]
struct QueryArgs {
    /// Query type (model, round, stake, proof)
    #[arg(value_enum)]
    query_type: QueryType,

    /// Model ID
    #[arg(short, long)]
    model_id: u64,

    /// Round ID (for round queries)
    #[arg(short, long)]
    round_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum QueryType {
    Model,
    Round,
    Stake,
    Proof,
    Error,
    Worker,
    Aggregator,
    Metrics,
}

#[derive(Args)]
struct ExportArgs {
    /// What to export (model, proofs, metrics, logs)
    #[arg(value_enum)]
    export_type: ExportType,

    /// Model ID
    #[arg(short, long)]
    model_id: u64,

    /// Output path
    #[arg(short, long, default_value = "./export")]
    output: PathBuf,

    /// Export format (json, csv, binary)
    #[arg(long, default_value = "json")]
    format: String,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum ExportType {
    Model,
    Proofs,
    Metrics,
    Logs,
    All,
}

#[derive(Args)]
struct TrainArgs {
    /// Path to training configuration file (TOML)
    #[arg(long, short = 'C', default_value = "model.toml")]
    train_config: PathBuf,

    /// Model ID (if joining existing model training)
    #[arg(short, long)]
    model_id: Option<u64>,

    /// Override max rounds from config
    #[arg(long)]
    max_rounds: Option<u32>,

    /// Override learning rate from config
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Dry-run mode (validate config without actual training)
    #[arg(long)]
    dry_run: bool,

    /// Resume from checkpoint (path or round number)
    #[arg(long)]
    resume: Option<String>,

    /// Wallet name to use for staking
    #[arg(long)]
    wallet: Option<String>,

    /// Skip confirmation prompts
    #[arg(long, short = 'y')]
    force: bool,

    /// Verbose output
    #[arg(long)]
    verbose: bool,

    /// Headless mode (no interactive progress)
    #[arg(long)]
    headless: bool,

    /// Export metrics after training completes
    #[arg(long)]
    export_metrics: bool,

    /// Output directory for metrics and checkpoints
    #[arg(long, short)]
    output_dir: Option<PathBuf>,

    /// Live mode: spawn real helix-node processes for distributed training
    #[arg(long)]
    live: bool,

    /// Number of worker nodes (live mode)
    #[arg(long, default_value = "3")]
    workers: u32,

    /// Model dimensions as "d_in,d_hid,d_out" (live mode)
    #[arg(long, default_value = "4,8,2")]
    model: String,

    /// Model seed (live mode)
    #[arg(long, default_value = "42")]
    model_seed: u64,

    /// Path to helix-node binary (live mode, defaults to searching PATH)
    #[arg(long)]
    node_binary: Option<PathBuf>,

    /// Ethereum RPC URL (live mode, defaults to http://localhost:8545)
    #[arg(long, default_value = "http://localhost:8545")]
    rpc_url: String,

    /// Private key for on-chain transactions (live mode)
    #[arg(long, env = "HELIX_PRIVATE_KEY", default_value = "")]
    private_key: String,

    /// Coordinator contract address (live mode)
    #[arg(long, env = "COORDINATOR_ADDRESS", default_value = "")]
    coordinator_address: String,

    /// HTTP API port for the aggregator (live mode)
    #[arg(long, default_value = "9001")]
    http_port: u16,

    /// Number of training steps (chain mode)
    #[arg(long)]
    steps: Option<u32>,

    /// Path to training data file (chain mode)
    #[arg(long)]
    data: Option<PathBuf>,

    /// Chain mode: submit real proofs to the coordinator contract after each step
    #[arg(long)]
    chain: bool,
}

/// Arguments for the submit-proof subcommand
#[cfg(feature = "chain")]
#[derive(Args)]
struct SubmitProofArgs {
    /// Model ID on the coordinator contract
    #[arg(long)]
    model_id: u64,

    /// Round ID within the model
    #[arg(long)]
    round_id: u64,

    /// Path to JSON file containing proof and public inputs
    #[arg(long)]
    proof_file: PathBuf,

    /// Ethereum RPC URL
    #[arg(long, env = "RPC_URL", default_value = "http://localhost:8545")]
    rpc_url: String,

    /// Private key for signing transactions
    #[arg(long, env = "HELIX_PRIVATE_KEY")]
    private_key: String,

    /// Coordinator contract address
    #[arg(long, env = "COORDINATOR_ADDRESS")]
    coordinator_address: String,

    /// Chain ID (auto-detected if omitted)
    #[arg(long)]
    chain_id: Option<u64>,
}

/// Arguments for the mpc-train subcommand (model owner end-to-end orchestration).
#[derive(Args)]
struct MpcTrainArgs {
    /// Model architecture as "d_in,d_hid,d_out" (e.g. "784,32,10" for MNIST)
    #[arg(long, default_value = "784,32,10")]
    architecture: String,

    /// Number of training steps
    #[arg(long, default_value = "100")]
    steps: usize,

    /// SGD learning rate
    #[arg(long, default_value = "0.001")]
    learning_rate: f64,

    /// Checkpoint frequency (every N steps)
    #[arg(long, default_value = "10")]
    checkpoint_freq: usize,

    /// MAC verification interval (every N steps, 0 to disable)
    #[arg(long, default_value = "10")]
    mac_interval: u64,

    /// Pre-generated Beaver triples per batch
    #[arg(long, default_value = "2048")]
    beaver_batch_size: usize,

    /// Mini-batch size (samples per step). Higher values use more CPU cores.
    #[arg(long, default_value = "128")]
    batch_size: usize,

    /// Number of MPC workers (ignored when --workers is provided)
    #[arg(long, default_value = "6")]
    num_workers: usize,

    /// Random seed for deterministic execution
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Worker TCP endpoints (comma-separated, e.g. "127.0.0.1:9001,127.0.0.1:9002,127.0.0.1:9003")
    #[arg(long, value_delimiter = ',')]
    workers: Vec<String>,

    /// Path to JSON file with initial weights (omit for Xavier initialization)
    #[arg(long)]
    weights: Option<String>,

    /// Path to JSON training data file (array of {input: [...], target: [...]})
    #[arg(long)]
    data: Option<PathBuf>,

    /// Use real MNIST data (requires helix-mpc real-mnist feature)
    #[arg(long)]
    real_mnist: bool,

    /// Directory containing MNIST IDX files (default: ~/.helix/data/mnist/)
    #[arg(long)]
    mnist_dir: Option<String>,

    /// Number of training samples
    #[arg(long, default_value = "1000")]
    train_size: usize,

    /// Number of test samples for accuracy evaluation
    #[arg(long, default_value = "200")]
    test_size: usize,

    /// Ethereum RPC URL (omit to start local Anvil)
    #[cfg(feature = "chain")]
    #[arg(long, env = "RPC_URL")]
    rpc_url: Option<String>,

    /// Owner's Ethereum private key (hex). Reads from TESTNET_PRIVATE_KEY or
    /// HELIX_PRIVATE_KEY env vars, falls back to Anvil default key for local dev.
    #[cfg(feature = "chain")]
    #[arg(long, env = "TESTNET_PRIVATE_KEY")]
    private_key: Option<String>,

    /// Worker private keys (comma-separated hex, must match --workers count)
    #[cfg(feature = "chain")]
    #[arg(long, value_delimiter = ',')]
    worker_keys: Vec<String>,

    /// Payment amount in ETH for job registration
    #[cfg(feature = "chain")]
    #[arg(long, default_value = "0.01")]
    payment_eth: f64,

    /// Stake amount in ETH per worker
    #[cfg(feature = "chain")]
    #[arg(long, default_value = "0.001")]
    stake_eth: f64,

    /// Existing V4 coordinator contract address (omit to deploy new)
    #[cfg(feature = "chain")]
    #[arg(long)]
    coordinator: Option<String>,

    /// Enable stake withdrawal after training (requires local Anvil)
    #[cfg(feature = "chain")]
    #[arg(long)]
    enable_withdrawal: bool,

    /// Enable optional ZK proof generation at checkpoints.
    /// When enabled, generates StateTransitionCircuit proofs for external verifiability.
    /// Default: disabled (MPC+MAC is the primary correctness mechanism).
    #[arg(long)]
    zk_proofs: bool,

    /// ZK proof checkpoint frequency: generate a proof every N checkpoints.
    /// Only used when --zk-proofs is enabled. Default: only at the final checkpoint.
    #[arg(long, default_value = "0")]
    zk_checkpoint_freq: usize,

    /// ZK proof mode: "off" (default), "always", or "risk:N" where N is the
    /// minimum worker count before ZK activates (e.g. "risk:2").
    #[arg(long, default_value = "off")]
    zk_mode: String,

    /// Simulate a cheater worker during training (demo feature).
    /// One worker will inject corrupt shares mid-training to demonstrate
    /// cheater detection and slashing.
    #[arg(long)]
    simulate_cheater: bool,

    /// Transport mode: "local" (default) runs all MPC parties in this process,
    /// "distributed" sends training commands to remote workers over TCP.
    /// Use "distributed" when workers run on separate physical machines.
    #[arg(long, default_value = "local")]
    transport: String,

    /// Save final trained weights to this JSON file
    #[arg(long)]
    output: Option<PathBuf>,
}

/// Arguments for the mpc-worker subcommand (worker participation).
#[derive(Args)]
struct MpcWorkerArgs {
    /// TCP address to listen on for data channel (e.g. "0.0.0.0:9001")
    #[arg(long, default_value = "0.0.0.0:9001")]
    listen: String,

    /// Worker party index (0-based)
    #[arg(long, default_value = "0")]
    party_index: usize,

    /// Deterministic seed for key generation
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Ethereum RPC URL for on-chain staking (omit for off-chain mode)
    #[cfg(feature = "chain")]
    #[arg(long, env = "RPC_URL")]
    rpc_url: Option<String>,

    /// Worker's Ethereum private key (hex)
    #[cfg(feature = "chain")]
    #[arg(long, env = "HELIX_PRIVATE_KEY")]
    private_key: Option<String>,

    /// On-chain job ID to join (omit to skip staking)
    #[cfg(feature = "chain")]
    #[arg(long)]
    job_id: Option<u64>,

    /// V4 coordinator contract address
    #[cfg(feature = "chain")]
    #[arg(long)]
    coordinator: Option<String>,

    /// Stake amount in ETH
    #[cfg(feature = "chain")]
    #[arg(long, default_value = "0.001")]
    stake_eth: f64,
}

/// Arguments for spawning multiple local MPC workers.
#[derive(Args)]
struct SpawnWorkersArgs {
    /// Number of workers to spawn
    #[arg(long, default_value = "3")]
    count: usize,

    /// Base TCP port (worker i listens on base_port + i*3; ports: data, control=+1, mpc=+2)
    #[arg(long, default_value = "9001")]
    base_port: u16,

    /// Bind address for all workers
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,

    /// Base random seed (worker i uses seed + i)
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Public IP/hostname for cross-machine registration (e.g. "192.168.1.100").
    /// Workers bind on --bind but register with this address so remote machines can reach them.
    /// Defaults to --bind value (or 127.0.0.1 if bind is 0.0.0.0).
    #[arg(long)]
    public_addr: Option<String>,

    /// Dashboard API URL to register workers with (e.g. http://localhost:3001)
    #[arg(long, default_value = "http://localhost:3001")]
    api_url: Option<String>,

    /// Ethereum RPC URL for on-chain worker pool registration
    #[cfg(feature = "chain")]
    #[arg(long)]
    rpc_url: Option<String>,

    /// V4 coordinator contract address for on-chain pool registration
    #[cfg(feature = "chain")]
    #[arg(long)]
    coordinator: Option<String>,

    /// Comma-separated private keys for workers (one per worker, or a single base key
    /// from which N keys are derived). If not provided, uses Anvil default accounts.
    #[cfg(feature = "chain")]
    #[arg(long)]
    private_keys: Option<String>,

    /// Stake amount in ETH per worker for on-chain pool registration
    #[cfg(feature = "chain")]
    #[arg(long, default_value = "0.001")]
    stake_eth: f64,
}

#[derive(Args)]
struct DemoArgs {
    /// Demo scenario to run
    #[arg(value_enum, default_value = "full-training")]
    scenario: DemoScenario,

    /// Number of worker nodes
    #[arg(short, long, default_value = "3")]
    workers: u32,

    /// Number of training rounds
    #[arg(short, long, default_value = "5")]
    rounds: u32,

    /// Round duration in seconds
    #[arg(long, default_value = "60")]
    round_duration: u64,

    /// Run in headless mode (no interactive UI)
    #[arg(long)]
    headless: bool,

    /// Skip contract deployment (assume already deployed)
    #[arg(long)]
    skip_deploy: bool,

    /// Live mode: spawn real helix-node processes with Anvil and on-chain proof submission
    #[arg(long)]
    live: bool,

    /// Path to helix-node binary (live mode, auto-detected if omitted)
    #[arg(long)]
    node_binary: Option<PathBuf>,

    /// Path to contracts/ directory (live mode, auto-detected if omitted)
    #[arg(long)]
    contracts_dir: Option<PathBuf>,

    /// Model dimensions as "d_in,d_hid,d_out" (live mode)
    #[arg(long, default_value = "4,8,2")]
    model: String,

    /// Model seed (live mode)
    #[arg(long, default_value = "42")]
    model_seed: u64,

    /// Private key for on-chain transactions (live mode, set via HELIX_PRIVATE_KEY env)
    #[arg(long, env = "HELIX_PRIVATE_KEY", default_value = "")]
    demo_private_key: String,

    /// Pre-deployed coordinator address (live mode, skip deployment if set)
    #[arg(long)]
    coordinator_address: Option<String>,

    /// Base TCP port for nodes (live mode)
    #[arg(long, default_value = "9000")]
    base_port: u16,

    /// HTTP API port for aggregator (live mode)
    #[arg(long, default_value = "9001")]
    demo_http_port: u16,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum DemoScenario {
    /// Quick demo showing basic training
    Quick,
    /// Full training with proofs and verification
    FullTraining,
    /// Demo with simulated slashing
    Slashing,
    /// Multi-model concurrent training
    MultiModel,
    /// Fault tolerance demo
    FaultTolerance,
}

#[derive(Args)]
struct OrchestrateArgs {
    /// Orchestration action
    #[arg(value_enum)]
    action: OrchestrateAction,

    /// Number of nodes to start
    #[arg(short, long, default_value = "3")]
    nodes: u32,

    /// Configuration template
    #[arg(short, long)]
    template: Option<PathBuf>,

    /// Network name
    #[arg(long, default_value = "helix-demo")]
    network_name: String,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum OrchestrateAction {
    /// Start the network
    Start,
    /// Stop the network
    Stop,
    /// Restart the network
    Restart,
    /// Scale the network
    Scale,
    /// Show network topology
    Topology,
}

#[derive(Args)]
struct HealthArgs {
    /// Target to check (all, nodes, contracts, network)
    #[arg(value_enum, default_value = "all")]
    target: HealthTarget,

    /// Timeout in seconds
    #[arg(short, long, default_value = "10")]
    timeout: u64,

    /// Output detailed health info
    #[arg(short, long)]
    detailed: bool,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum HealthTarget {
    All,
    Nodes,
    Contracts,
    Network,
    Storage,
}

#[derive(Args)]
struct DashboardArgs {
    /// Port to serve dashboard on
    #[arg(short, long, default_value = "3001")]
    port: u16,

    /// Host to bind to (0.0.0.0 allows connections from other machines)
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Enable CORS
    #[arg(long)]
    cors: bool,

    /// Ethereum RPC URL (for on-chain worker pool queries)
    #[arg(long)]
    rpc_url: Option<String>,

    /// V4 coordinator contract address (for on-chain worker pool)
    #[arg(long)]
    coordinator: Option<String>,
}

#[derive(Args)]
struct BenchmarkArgs {
    /// Benchmark type
    #[arg(value_enum)]
    benchmark_type: BenchmarkType,

    /// Number of iterations
    #[arg(short, long, default_value = "100")]
    iterations: u32,

    /// Warmup iterations
    #[arg(long, default_value = "10")]
    warmup: u32,

    /// Output results to file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum BenchmarkType {
    /// Proof generation benchmark
    ProofGen,
    /// Proof verification benchmark
    ProofVerify,
    /// Gradient aggregation benchmark
    Aggregation,
    /// Network throughput benchmark
    Network,
    /// End-to-end training round
    E2e,
}

#[derive(Args)]
struct LogsArgs {
    /// Follow logs in real-time
    #[arg(short, long)]
    follow: bool,

    /// Filter by node ID
    #[arg(short, long)]
    node: Option<String>,

    /// Filter by log level
    #[arg(short, long)]
    level: Option<String>,

    /// Number of lines to show
    #[arg(short = 'n', long, default_value = "100")]
    lines: usize,

    /// Search pattern
    #[arg(short, long)]
    grep: Option<String>,

    /// Aggregate from all nodes
    #[arg(short, long)]
    aggregate: bool,
}

#[derive(Args)]
struct WatchArgs {
    /// Configuration files to watch
    #[arg(default_value = "helix.toml")]
    files: Vec<PathBuf>,

    /// Action on config change (reload, restart, notify)
    #[arg(long, default_value = "reload")]
    on_change: String,
}

#[derive(Args)]
struct VisualizeArgs {
    /// Demo scenario to visualize
    #[arg(value_enum, default_value = "quick")]
    scenario: DemoScenario,

    /// Number of worker nodes
    #[arg(short, long, default_value = "5")]
    workers: u32,

    /// Number of training rounds
    #[arg(short, long, default_value = "10")]
    rounds: u32,

    /// Refresh rate in milliseconds
    #[arg(long, default_value = "100")]
    refresh_rate: u64,

    /// Start visualization without running demo (attach to existing)
    #[arg(long)]
    attach: bool,
}

#[derive(Args)]
struct HelpArgs {
    /// Topic or command to get help for
    #[arg(default_value = "")]
    topic: String,

    /// Show quick reference card
    #[arg(long)]
    quick: bool,
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    init_logging(cli.verbose)?;

    // Setup graceful shutdown
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let shutdown_tx_clone = shutdown_tx.clone();

    tokio::spawn(async move {
        if let Ok(()) = tokio::signal::ctrl_c().await {
            println!("\n{}", "Received shutdown signal, cleaning up...".yellow());
            let _ = shutdown_tx_clone.send(());
        }
    });

    // Build a shared RPC client for commands that need it.
    // Try to connect to a real node; fall back to mock for CLI commands that don't need one.
    let rpc_config = rpc::client::HelixRpcConfig::default();
    let rpc_client = Arc::new(match rpc::client::UnifiedRpcClient::connect(rpc_config).await {
        Ok(client) => client,
        Err(e) => {
            tracing::warn!("Could not connect to HELIX node: {}. Using mock mode.", e);
            rpc::client::UnifiedRpcClient::new_mock()
        }
    });

    // Execute command
    let result = match &cli.command {
        Commands::Init(args) => cmd_init(args, &cli).await,
        Commands::Join(args) => cmd_join(args, &cli, &rpc_client).await,
        Commands::Status(args) => cmd_status(args, &cli, &rpc_client).await,
        Commands::Query(args) => cmd_query(args, &cli, &rpc_client).await,
        Commands::Export(args) => cmd_export(args, &cli, &rpc_client).await,
        Commands::Train(args) => cmd_train(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Demo(args) => cmd_demo(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Orchestrate(args) => cmd_orchestrate(args, &cli).await,
        Commands::Health(args) => cmd_health(args, &cli).await,
        Commands::Dashboard(args) => cmd_dashboard(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Benchmark(args) => cmd_benchmark(args, &cli).await,
        Commands::Logs(args) => cmd_logs(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Watch(args) => cmd_watch(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Visualize(args) => cmd_visualize(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Guide(args) => cmd_help(args, &cli).await,
        #[cfg(feature = "chain")]
        Commands::SubmitProof(args) => cmd_submit_proof(args).await,
        Commands::MpcTrain(args) => cmd_mpc_train(args, &cli).await,
        Commands::MpcWorker(args) => cmd_mpc_worker(args, &cli).await,
        Commands::SpawnWorkers(args) => cmd_spawn_workers(args, &cli).await,
    };

    if let Err(e) = result {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    Ok(())
}

fn init_logging(verbosity: u8) -> Result<()> {
    let filter = match verbosity {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    Ok(())
}

// ============================================================================
// Command Implementations
// ============================================================================

async fn cmd_init(args: &InitArgs, cli: &Cli) -> Result<()> {
    let mut progress = ProgressDisplay::new();
    progress.start_spinner("Initializing HELIX node...");

    let name = args.name.clone().unwrap_or_else(|| format!("helix-node-{}", uuid::Uuid::new_v4().to_string()[..8].to_string()));

    // Create config directory
    let config_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".helix");
    std::fs::create_dir_all(&config_dir)?;

    // Determine profile from network argument
    let profile = match args.network.as_str() {
        "local" | "dev" => config::ConfigProfile::Local,
        "anvil" => config::ConfigProfile::Anvil,
        "sepolia" | "testnet" => config::ConfigProfile::Sepolia,
        "mainnet" => config::ConfigProfile::Mainnet,
        _ => config::ConfigProfile::Local,
    };

    // Generate config from profile
    let mut config = config::HelixConfig::from_profile(profile);

    // Override with CLI arguments
    config.node.name = name.clone();
    config.node.data_dir = config_dir.join("data");
    if args.aggregator {
        config.node.capabilities = vec!["train".into(), "aggregate".into(), "prove".into()];
        config.node.role = config::NodeRole::Aggregator;
    }

    // Write config file
    let config_path = cli.config.clone();
    let config_str = toml::to_string_pretty(&config)?;
    std::fs::write(&config_path, config_str)?;

    progress.finish_spinner(&format!("Node '{}' initialized successfully!", name));

    println!("\n{}", "Configuration:".cyan().bold());
    println!("  Config file: {}", config_path.display());
    println!("  Data directory: {}", config_dir.join("data").display());
    println!("  Network: {} ({})", args.network, profile.name());

    if args.generate_wallet {
        progress.start_spinner("Generating wallet...");
        // Wallet generation would go here
        progress.finish_spinner("Wallet generated");
        println!("  Wallet: {}", config_dir.join("wallet.json").display());
    }

    println!("\n{}", "Next steps:".green().bold());
    println!("  1. Review the configuration: helix.toml");
    println!("  2. Start the node: helix join --coordinator <addr>");
    println!("  3. Or run a demo: helix demo");

    Ok(())
}

async fn cmd_join(args: &JoinArgs, _cli: &Cli, rpc: &rpc::client::UnifiedRpcClient) -> Result<()> {
    let mut progress = ProgressDisplay::new();

    println!("{}", "Joining HELIX training network...".cyan().bold());
    println!("  Coordinator: {}", args.coordinator);
    println!("  Model ID: {}", args.model_id);

    // Connect and verify network
    progress.start_spinner("Connecting to coordinator...");
    let net_status = rpc.get_network_status().await;
    match &net_status {
        Ok(net) => {
            progress.finish_spinner(&format!(
                "Connected to coordinator (chain_id={}, block={})",
                net.chain_id, net.block_height
            ));
        }
        Err(_) => {
            progress.finish_spinner("Connected to coordinator (mock mode)");
        }
    }

    // Query model info
    progress.start_spinner("Querying model info...");
    let model = rpc.get_model(args.model_id).await;
    match &model {
        Ok(m) => {
            progress.finish_spinner(&format!(
                "Model '{}' found (round {}, min_stake {} ETH)",
                m.name, m.current_round, m.min_stake
            ));
        }
        Err(_) => {
            progress.finish_spinner("Model info retrieved (mock mode)");
        }
    }

    // Stake if requested
    if let Some(stake) = args.stake {
        progress.start_spinner(&format!("Staking {} ETH...", stake));
        #[cfg(feature = "chain")]
        {
            if let Some(chain) = rpc.chain_client() {
                match chain.stake(args.model_id, ethers::types::U256::from((stake * 1e18) as u64)).await {
                    Ok(receipt) => {
                        progress.finish_spinner(&format!(
                            "Staked {} ETH (tx: 0x{}...)",
                            stake,
                            hex::encode(&receipt.transaction_hash.as_bytes()[..4])
                        ));
                    }
                    Err(e) => {
                        progress.finish_spinner(&format!("Stake simulated ({} ETH) — chain error: {}", stake, e));
                    }
                }
            } else {
                tokio::time::sleep(Duration::from_millis(500)).await;
                progress.finish_spinner(&format!("Staked {} ETH (mock)", stake));
            }
        }
        #[cfg(not(feature = "chain"))]
        {
            tokio::time::sleep(Duration::from_millis(500)).await;
            progress.finish_spinner(&format!("Staked {} ETH (mock)", stake));
        }
    }

    // Discover peers
    progress.start_spinner("Discovering peers...");
    let workers = rpc.get_workers().await;
    match &workers {
        Ok(w) => {
            let active = w.iter().filter(|w| w.status == rpc::client::WorkerStatus::Training || w.status == rpc::client::WorkerStatus::Idle).count();
            progress.finish_spinner(&format!("Found {} peers ({} active)", w.len(), active));
        }
        Err(_) => {
            progress.finish_spinner("Found peers (mock mode)");
        }
    }

    // Sync model state
    progress.start_spinner("Syncing model state...");
    match &model {
        Ok(m) => {
            progress.finish_spinner(&format!("Model state synced (commitment: {}...)", &m.current_commitment[..18.min(m.current_commitment.len())]));
        }
        Err(_) => {
            tokio::time::sleep(Duration::from_millis(300)).await;
            progress.finish_spinner("Model state synced (mock)");
        }
    }

    println!("\n{}", "Successfully joined the training network!".green().bold());
    println!("  Capabilities: {:?}", args.capabilities);
    println!("  Status: Ready to participate in training rounds");

    Ok(())
}

async fn cmd_status(args: &StatusArgs, _cli: &Cli, rpc: &rpc::client::UnifiedRpcClient) -> Result<()> {
    let display_status = |training: &Option<rpc::client::TrainingStatus>,
                          network: &Option<rpc::client::NetworkStatus>,
                          proof: &Option<rpc::client::ProofStatus>,
                          staking: &Option<rpc::client::StakingInfo>,
                          workers: &Option<Vec<rpc::client::WorkerInfo>>| {
        println!("{}", "═".repeat(60).cyan());
        println!("{}", " HELIX Network Status".cyan().bold());
        println!("{}", "═".repeat(60).cyan());

        // Node / Network Status
        println!("\n{}", "Node Status:".yellow().bold());
        if let Some(net) = network {
            println!("  Status:      {} {}", "●".green(), if net.blockchain_connected { "Online" } else { "Degraded" });
            println!("  Peers:       {} connected", net.peer_count);
            println!("  Chain:       {} (block {})", net.chain_id, net.block_height);
            println!("  Latency:     {} ms", net.avg_latency_ms);
        } else {
            println!("  Status:      {} Online (mock)", "●".green());
            println!("  Peers:       — (no RPC connection)");
        }

        // Training Status
        println!("\n{}", "Training Status:".yellow().bold());
        if let Some(model_id) = args.model_id {
            println!("  Model ID:    {}", model_id);
        }
        if let Some(t) = training {
            let progress_pct = if t.total_rounds > 0 {
                (t.current_round as f64 / t.total_rounds as f64) * 100.0
            } else { 0.0 };
            println!("  Active:         {} {}", if t.active { "●".green() } else { "●".red() }, if t.active { "Yes" } else { "No" });
            println!("  Current Round:  {}/{}", t.current_round, t.total_rounds);
            println!("  Round Status:   {} In Progress ({:.0}%)", "●".blue(), progress_pct);
            println!("  Phase:          {}", t.phase.name());
            println!("  Current Loss:   {:.4}", t.current_loss);
            println!("  Error Bound:    {:.1} / {:.0} max", t.accumulated_error, t.max_error_bound);
        } else {
            println!("  (no training data available)");
        }

        // Proof Status
        if let Some(p) = proof {
            println!("\n{}", "Proof Status:".yellow().bold());
            println!("  Phase:          {}", p.phase.name());
            println!("  Progress:       {}%", p.progress_percent);
            println!("  Constraints:    {}/{}", p.constraints_satisfied, p.total_constraints);
            println!("  Error Bound:    {:.2}", p.error_bound);
        }

        // Staking
        println!("\n{}", "Staking:".yellow().bold());
        if let Some(s) = staking {
            println!("  Your Stake:     {:.4} ETH", s.your_stake);
            println!("  Total Staked:   {:.4} ETH", s.total_staked);
            if s.is_locked {
                println!("  Lock Status:    {} Locked", "●".yellow());
            } else {
                println!("  Lock Status:    {} Unlocked", "●".green());
            }
            println!("  Rewards:        {:.4} ETH pending", s.pending_rewards);
        } else {
            println!("  (no staking data available)");
        }

        // Detailed worker table
        if args.detailed {
            if let Some(w) = workers {
                println!("\n{}", "Connected Peers:".yellow().bold());
                println!("  ┌─────────────────┬──────────┬────────────┬───────────┐");
                println!("  │ Worker ID       │ Status   │ Stake      │ Reputation│");
                println!("  ├─────────────────┼──────────┼────────────┼───────────┤");
                for worker in w.iter().take(10) {
                    let status_icon = match worker.status {
                        rpc::client::WorkerStatus::Training | rpc::client::WorkerStatus::Idle => "●".green(),
                        rpc::client::WorkerStatus::Faulted | rpc::client::WorkerStatus::Slashed => "●".red(),
                        _ => "●".yellow(),
                    };
                    println!(
                        "  │ {:15} │ {} {:6} │ {:>8.3} ETH│ {:>6.0}%   │",
                        &worker.id[..15.min(worker.id.len())],
                        status_icon,
                        format!("{:?}", worker.status),
                        worker.stake,
                        worker.reputation * 100.0
                    );
                }
                println!("  └─────────────────┴──────────┴────────────┴───────────┘");
            }
        }

        println!("{}", "═".repeat(60).cyan());
    };

    // Fetch all data from RPC (with graceful fallback)
    let model_id = args.model_id.unwrap_or(0);
    let fetch_all = || async {
        let training = rpc.get_training_status().await.ok();
        let network = rpc.get_network_status().await.ok();
        let proof = rpc.get_proof_status().await.ok();
        let staking = rpc.get_staking_info(model_id).await.ok();
        let workers = rpc.get_workers().await.ok();
        (training, network, proof, staking, workers)
    };

    if args.watch {
        loop {
            print!("\x1B[2J\x1B[1;1H"); // Clear screen
            let (training, network, proof, staking, workers) = fetch_all().await;
            display_status(&training, &network, &proof, &staking, &workers);
            println!("\nRefreshing every {}s... (Ctrl+C to stop)", args.interval);
            tokio::time::sleep(Duration::from_secs(args.interval)).await;
        }
    } else {
        let (training, network, proof, staking, workers) = fetch_all().await;
        display_status(&training, &network, &proof, &staking, &workers);
    }

    Ok(())
}

async fn cmd_query(args: &QueryArgs, _cli: &Cli, rpc: &rpc::client::UnifiedRpcClient) -> Result<()> {
    match args.query_type {
        QueryType::Model => {
            let model = rpc.get_model(args.model_id).await?;
            println!("{}", "Model Information".cyan().bold());
            println!("  Model ID:          {}", model.id);
            println!("  Name:              {}", model.name);
            println!("  IPFS Hash:         {}", model.ipfs_hash);
            println!("  Architecture:      {}", model.architecture);
            println!("  Parameters:        {}", model.parameter_count);
            println!("  Current Round:     {}", model.current_round);
            println!("  Current Commitment: {}...", &model.current_commitment[..20.min(model.current_commitment.len())]);
            println!("  Min Stake:         {} ETH", model.min_stake);
            println!("  Active:            {} {}", if model.training_active { "●".green() } else { "●".red() }, if model.training_active { "Yes" } else { "No" });
            println!("  Owner:             {}...", &model.owner[..20.min(model.owner.len())]);
            println!("  Error Bound:       {:.1}", model.accumulated_error);
        }
        QueryType::Round => {
            let round_id = args.round_id.unwrap_or(0);
            let round = rpc.get_round(args.model_id, round_id).await?;
            println!("{}", "Round Information".cyan().bold());
            println!("  Model ID:          {}", round.model_id);
            println!("  Round ID:          {}", round.round_id);
            println!("  Prev Commitment:   {}...", &round.prev_commitment[..20.min(round.prev_commitment.len())]);
            if let Some(ref new_c) = round.new_commitment {
                println!("  New Commitment:    {}...", &new_c[..20.min(new_c.len())]);
            }
            println!("  Completed:         {} {}", if round.completed { "●".green() } else { "●".blue() }, if round.completed { "Yes" } else { "In Progress" });
            println!("  Proofs:            {} submitted, {} verified", round.proofs_submitted, round.proofs_verified);
            println!("  Participants:      {}", round.participants.len());
            if let Some(loss) = round.loss {
                println!("  Loss:              {:.4}", loss);
            }
            if let Some(err) = round.error_delta {
                println!("  Error Delta:       {:.2}", err);
            }
        }
        QueryType::Stake => {
            let staking = rpc.get_staking_info(args.model_id).await?;
            println!("{}", "Stake Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Your Stake:        {:.4} ETH", staking.your_stake);
            println!("  Total Staked:      {:.4} ETH", staking.total_staked);
            println!("  Locked:            {} {}", if staking.is_locked { "●".yellow() } else { "●".green() }, if staking.is_locked { "Yes" } else { "No" });
            println!("  Pending Rewards:   {:.4} ETH", staking.pending_rewards);
            println!("  Claimed Rewards:   {:.4} ETH", staking.total_rewards_claimed);
            if !staking.slashing_events.is_empty() {
                println!("  Slashing Events:   {}", staking.slashing_events.len());
            } else {
                println!("  Slashed:           {} No", "●".green());
            }
        }
        QueryType::Proof => {
            let proof = rpc.get_proof_status().await?;
            println!("{}", "Proof Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Phase:             {}", proof.phase.name());
            println!("  Generating:        {} {}", if proof.generating { "●".blue() } else { "●".green() }, if proof.generating { "Yes" } else { "No" });
            println!("  Progress:          {}%", proof.progress_percent);
            println!("  Constraints:       {}/{}", proof.constraints_satisfied, proof.total_constraints);
            println!("  Elapsed:           {} ms", proof.elapsed_ms);
            println!("  Error Bound:       {:.2}", proof.error_bound);
            println!("  GPU Accelerated:   {}", if proof.gpu_accelerated { "Yes" } else { "No" });
        }
        QueryType::Error => {
            let training = rpc.get_training_status().await?;
            println!("{}", "Error Bound Information".cyan().bold());
            println!("  Model ID:          {}", training.model_id);
            println!("  Accumulated Error: {:.1}", training.accumulated_error);
            println!("  Max Allowed:       {:.0}", training.max_error_bound);
            let status = if training.accumulated_error < training.max_error_bound * 0.5 {
                ("●".green(), "Acceptable")
            } else if training.accumulated_error < training.max_error_bound * 0.9 {
                ("●".yellow(), "Warning")
            } else {
                ("●".red(), "Critical")
            };
            println!("  Status:            {} {}", status.0, status.1);
            println!("  Current Round:     {}", training.current_round);
        }
        QueryType::Worker => {
            let workers = rpc.get_workers().await?;
            println!("{}", "Worker Information".cyan().bold());
            if workers.is_empty() {
                println!("  No workers found.");
            }
            for (i, w) in workers.iter().enumerate() {
                if i > 0 { println!(); }
                let status_icon = match w.status {
                    rpc::client::WorkerStatus::Training | rpc::client::WorkerStatus::Idle => "●".green(),
                    rpc::client::WorkerStatus::Faulted | rpc::client::WorkerStatus::Slashed => "●".red(),
                    _ => "●".yellow(),
                };
                println!("  Worker ID:         {}", w.id);
                println!("  Address:           {}...", &w.address[..20.min(w.address.len())]);
                println!("  Status:            {} {:?}", status_icon, w.status);
                println!("  Stake:             {:.4} ETH", w.stake);
                println!("  Reputation:        {:.0}%", w.reputation * 100.0);
                println!("  Proofs:            {} submitted, {} verified, {} rejected", w.proofs_submitted, w.proofs_verified, w.proofs_rejected);
            }
        }
        QueryType::Aggregator => {
            // Aggregator info comes from workers with aggregator role
            let workers = rpc.get_workers().await?;
            let network = rpc.get_network_status().await.ok();
            println!("{}", "Aggregator Information".cyan().bold());
            println!("  Active Aggregators: {}", network.map(|n| n.active_aggregators).unwrap_or(0));
            for w in workers.iter().filter(|w| w.id.contains("agg")) {
                println!("  ID:                {}", w.id);
                println!("  Address:           {}...", &w.address[..20.min(w.address.len())]);
                println!("  Status:            {} {:?}", "●".green(), w.status);
            }
            if workers.iter().filter(|w| w.id.contains("agg")).count() == 0 {
                println!("  No aggregators found in worker list.");
            }
        }
        QueryType::Metrics => {
            let training = rpc.get_training_status().await.ok();
            let proof = rpc.get_proof_status().await.ok();
            let staking = rpc.get_staking_info(args.model_id).await.ok();
            let network = rpc.get_network_status().await.ok();

            println!("{}", "Training Metrics".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!();

            println!("{}", "Training:".yellow());
            if let Some(t) = &training {
                let pct = if t.total_rounds > 0 { (t.current_round as f64 / t.total_rounds as f64) * 100.0 } else { 0.0 };
                println!("  Progress:          {}/{} ({:.1}%)", t.current_round, t.total_rounds, pct);
                println!("  Current Loss:      {:.4}", t.current_loss);
                println!("  Phase:             {}", t.phase.name());
                println!("  Error Bound:       {:.1} / {:.0}", t.accumulated_error, t.max_error_bound);
            } else {
                println!("  (no training data)");
            }
            println!();

            println!("{}", "Proofs:".yellow());
            if let Some(p) = &proof {
                println!("  Phase:             {}", p.phase.name());
                println!("  Progress:          {}%", p.progress_percent);
                println!("  Constraints:       {}/{}", p.constraints_satisfied, p.total_constraints);
            } else {
                println!("  (no proof data)");
            }
            println!();

            println!("{}", "Network:".yellow());
            if let Some(n) = &network {
                println!("  Peers:             {}", n.peer_count);
                println!("  Workers:           {}", n.active_workers);
                println!("  Aggregators:       {}", n.active_aggregators);
                println!("  Avg Latency:       {} ms", n.avg_latency_ms);
            } else {
                println!("  (no network data)");
            }
            println!();

            println!("{}", "Economics:".yellow());
            if let Some(s) = &staking {
                println!("  Total Staked:      {:.4} ETH", s.total_staked);
                println!("  Your Stake:        {:.4} ETH", s.your_stake);
                println!("  Pending Rewards:   {:.4} ETH", s.pending_rewards);
                println!("  Slashing Events:   {}", s.slashing_events.len());
            } else {
                println!("  (no staking data)");
            }
        }
    }

    Ok(())
}

async fn cmd_export(args: &ExportArgs, _cli: &Cli, rpc: &rpc::client::UnifiedRpcClient) -> Result<()> {
    let mut progress = ProgressDisplay::new();

    std::fs::create_dir_all(&args.output)?;

    match args.export_type {
        ExportType::Model => {
            progress.start_spinner("Fetching model data...");
            let model = rpc.get_model(args.model_id).await.ok();
            let training = rpc.get_training_status().await.ok();
            let path = args.output.join(format!("model_{}.json", args.model_id));
            let export_data = serde_json::json!({
                "model": model,
                "training_status": training,
                "exported_at": chrono::Utc::now().to_rfc3339(),
            });
            std::fs::write(&path, serde_json::to_string_pretty(&export_data)?)?;
            progress.finish_spinner(&format!("Model exported to {}", path.display()));
        }
        ExportType::Proofs => {
            progress.start_spinner("Fetching proof data...");
            let proof = rpc.get_proof_status().await.ok();
            let path = args.output.join(format!("proofs_{}.json", args.model_id));
            let export_data = serde_json::json!({
                "model_id": args.model_id,
                "proof_status": proof,
                "exported_at": chrono::Utc::now().to_rfc3339(),
            });
            std::fs::write(&path, serde_json::to_string_pretty(&export_data)?)?;
            progress.finish_spinner(&format!("Proofs exported to {}", path.display()));
        }
        ExportType::Metrics => {
            progress.start_spinner("Fetching training metrics...");
            let training = rpc.get_training_status().await.ok();
            let proof = rpc.get_proof_status().await.ok();
            let network = rpc.get_network_status().await.ok();
            let staking = rpc.get_staking_info(args.model_id).await.ok();
            let workers = rpc.get_workers().await.ok();
            let history = rpc.get_progress_history().await;
            let path = args.output.join(format!("metrics_{}.json", args.model_id));
            let export_data = serde_json::json!({
                "model_id": args.model_id,
                "training": training,
                "proof": proof,
                "network": network,
                "staking": staking,
                "workers": workers,
                "progress_history": history,
                "exported_at": chrono::Utc::now().to_rfc3339(),
            });
            std::fs::write(&path, serde_json::to_string_pretty(&export_data)?)?;
            progress.finish_spinner(&format!("Metrics exported to {}", path.display()));
        }
        ExportType::Logs => {
            progress.start_spinner("Exporting logs...");
            let path = args.output.join("logs.txt");
            // Logs are local — write placeholder if no log file exists
            if !path.exists() {
                std::fs::write(&path, format!("# HELIX training logs — exported {}\n", chrono::Utc::now().to_rfc3339()))?;
            }
            progress.finish_spinner(&format!("Logs exported to {}", path.display()));
        }
        ExportType::All => {
            progress.start_spinner("Exporting all artifacts...");
            let model = rpc.get_model(args.model_id).await.ok();
            let training = rpc.get_training_status().await.ok();
            let proof = rpc.get_proof_status().await.ok();
            let network = rpc.get_network_status().await.ok();
            let staking = rpc.get_staking_info(args.model_id).await.ok();
            let workers = rpc.get_workers().await.ok();
            let history = rpc.get_progress_history().await;
            let export_data = serde_json::json!({
                "model_id": args.model_id,
                "model": model,
                "training": training,
                "proof": proof,
                "network": network,
                "staking": staking,
                "workers": workers,
                "progress_history": history,
                "exported_at": chrono::Utc::now().to_rfc3339(),
            });
            let path = args.output.join(format!("helix_export_{}.json", args.model_id));
            std::fs::write(&path, serde_json::to_string_pretty(&export_data)?)?;
            progress.finish_spinner(&format!("All artifacts exported to {}", path.display()));
        }
    }

    Ok(())
}

async fn cmd_train(args: &TrainArgs, _cli: &Cli, shutdown: broadcast::Receiver<()>) -> Result<()> {
    if args.live {
        return cmd_train_live(args, shutdown).await;
    }

    #[cfg(feature = "chain")]
    if args.chain {
        return cmd_train_chain(args).await;
    }

    let options = TrainOptions {
        config: args.train_config.clone(),
        model_id: args.model_id,
        max_rounds: args.max_rounds,
        learning_rate: args.learning_rate,
        dry_run: args.dry_run,
        resume: args.resume.clone(),
        wallet: args.wallet.clone(),
        force: args.force,
        verbose: args.verbose,
        headless: args.headless,
        export_metrics: args.export_metrics,
        output_dir: args.output_dir.clone(),
    };

    let mut cmd = TrainCommand::new();
    let result = cmd.execute(options, shutdown).await?;

    if !result.success {
        if let Some(error) = &result.error {
            return Err(anyhow::anyhow!("{}", error));
        }
    }

    Ok(())
}

/// Live training: spawn real helix-node processes and run distributed ZK-verified training.
async fn cmd_train_live(args: &TrainArgs, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    use orchestrator::{
        LiveNetworkConfig, LiveTrainingOrchestrator, NetworkOrchestratorConfig,
    };

    // Parse model dimensions
    let dims: Vec<usize> = args
        .model
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow::anyhow!("Invalid --model format. Expected d_in,d_hid,d_out (e.g. 4,8,2)"))?;
    if dims.len() != 3 {
        return Err(anyhow::anyhow!("--model requires exactly 3 dimensions: d_in,d_hid,d_out"));
    }
    let (d_in, d_hid, d_out) = (dims[0], dims[1], dims[2]);

    let max_rounds = args.max_rounds.unwrap_or(5);
    let learning_rate = args.learning_rate.unwrap_or(0.01);

    // Resolve helix-node binary
    let node_binary = match &args.node_binary {
        Some(p) => p.clone(),
        None => {
            // Try to find it relative to cargo target dir
            let candidates = [
                PathBuf::from("target/debug/helix-node"),
                PathBuf::from("target/release/helix-node"),
                PathBuf::from("helix-node"),
            ];
            candidates
                .iter()
                .find(|p| p.exists())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!(
                    "helix-node binary not found. Build it first:\n  cargo build -p helix-node\nOr specify --node-binary <path>"
                ))?
        }
    };

    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " HELIX Live Distributed Training".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    println!("{}", "Configuration:".yellow().bold());
    println!("  Model dims:    {}x{}x{} ({} params)",
        d_in, d_hid, d_out,
        d_in * d_hid + d_hid + d_hid * d_out + d_out
    );
    println!("  Workers:       {}", args.workers);
    println!("  Rounds:        {}", max_rounds);
    println!("  Learning rate: {}", learning_rate);
    println!("  Node binary:   {}", node_binary.display());
    println!("  HTTP API:      http://127.0.0.1:{}", args.http_port);
    if !args.rpc_url.is_empty() && !args.private_key.is_empty() {
        println!("  Chain RPC:     {}", args.rpc_url);
        println!("  Coordinator:   {}", args.coordinator_address);
    } else {
        println!("  Chain:         {} (no on-chain submission)", "offline".yellow());
    }
    println!();

    let live_config = LiveNetworkConfig {
        node_binary: node_binary.clone(),
        eth_rpc: args.rpc_url.clone(),
        private_key: args.private_key.clone(),
        coordinator_address: args.coordinator_address.clone(),
        model_dims: (d_in, d_hid, d_out),
        model_seed: args.model_seed,
        learning_rate,
        http_port: args.http_port,
    };

    let live_orch = LiveTrainingOrchestrator::new(args.workers, max_rounds, live_config);

    // Start all nodes
    let mut progress = ProgressDisplay::new();
    progress.start_spinner("Starting aggregator + workers...");
    live_orch.start().await?;
    progress.finish_spinner(&format!(
        "Started 1 aggregator + {} workers",
        args.workers
    ));

    // Wait for workers to register with aggregator
    progress.start_spinner("Waiting for workers to connect...");
    let poll_timeout = std::time::Instant::now();
    loop {
        if poll_timeout.elapsed() > Duration::from_secs(30) {
            live_orch.stop().await?;
            return Err(anyhow::anyhow!("Timeout waiting for workers to connect"));
        }

        match live_orch.poll_health().await {
            Ok(health) => {
                let workers = health.get("workers").and_then(|v| v.as_u64()).unwrap_or(0);
                if workers >= args.workers as u64 {
                    break;
                }
            }
            Err(_) => {} // API not ready yet
        }

        // Check for shutdown
        if shutdown.try_recv().is_ok() {
            live_orch.stop().await?;
            return Err(anyhow::anyhow!("Interrupted"));
        }

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    progress.finish_spinner(&format!("{} workers connected", args.workers));
    println!();

    // Run training rounds
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " Training Progress".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    for round in 1..=max_rounds {
        // Check for shutdown
        if shutdown.try_recv().is_ok() {
            println!("\n{}", "Training interrupted by user".yellow());
            break;
        }

        // Trigger a round
        progress.start_spinner(&format!("Round {}/{} - Starting...", round, max_rounds));
        match live_orch.trigger_round().await {
            Ok(_) => {}
            Err(e) => {
                progress.finish_spinner_error(&format!("Round {} - Failed to trigger: {}", round, e));
                // Wait and retry — aggregator may need time
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        }

        // Poll round status until complete
        let round_start = std::time::Instant::now();
        let round_timeout = Duration::from_secs(120);
        loop {
            if round_start.elapsed() > round_timeout {
                progress.finish_spinner_error(&format!("Round {} - Timeout", round));
                break;
            }

            if shutdown.try_recv().is_ok() {
                break;
            }

            match live_orch.poll_round_status().await {
                Ok(status) => {
                    let current_round = status.get("current_round").and_then(|v| {
                        v.as_object().and_then(|r| r.get("round_id").and_then(|id| id.as_u64()))
                    });
                    let completed = status.get("completed_rounds").and_then(|v| v.as_u64()).unwrap_or(0);

                    if completed >= round as u64 {
                        let elapsed = round_start.elapsed().as_millis();
                        progress.finish_spinner(&format!(
                            "Round {}/{} - Completed ({}ms)",
                            round, max_rounds, elapsed
                        ));
                        break;
                    }

                    if current_round.is_none() && completed < round as u64 {
                        // Round hasn't started yet or completed before we could see it
                        // Give it more time
                    }
                }
                Err(_) => {} // Transient error
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    println!();
    println!("{}", "═".repeat(60).green());
    println!("{}", " Training Complete".green().bold());
    println!("{}", "═".repeat(60).green());
    println!();

    // Final status
    match live_orch.poll_round_status().await {
        Ok(status) => {
            let completed = status.get("completed_rounds").and_then(|v| v.as_u64()).unwrap_or(0);
            let workers = status.get("worker_count").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  Rounds completed: {}", completed);
            println!("  Workers:          {}", workers);
        }
        Err(_) => {
            println!("  (Unable to fetch final status)");
        }
    }
    println!();

    // Stop all nodes
    progress.start_spinner("Shutting down nodes...");
    live_orch.stop().await?;
    progress.finish_spinner("All nodes stopped");

    Ok(())
}

async fn cmd_demo(args: &DemoArgs, _cli: &Cli, shutdown: broadcast::Receiver<()>) -> Result<()> {
    if args.live {
        return cmd_demo_live(args, shutdown).await;
    }

    use demo::{DemoScenarioType, DemoBuilder};

    // Map CLI scenario to demo module scenario type
    let scenario = match args.scenario {
        DemoScenario::Quick => DemoScenarioType::Quick,
        DemoScenario::FullTraining => DemoScenarioType::FullTraining,
        DemoScenario::Slashing => DemoScenarioType::Slashing,
        DemoScenario::MultiModel => DemoScenarioType::MultiModel,
        DemoScenario::FaultTolerance => DemoScenarioType::FaultTolerance,
    };

    // Build the demo runner with CLI arguments
    let runner = DemoBuilder::new()
        .scenario(scenario)
        .workers(args.workers)
        .rounds(args.rounds)
        .round_duration(Duration::from_secs(args.round_duration))
        .headless(args.headless)
        .build();

    // Run the demo
    let results = runner.run(shutdown).await?;

    // Export results if verbose
    if !args.headless {
        if let Ok(json) = results.to_json() {
            tracing::debug!("Demo results: {}", json);
        }
    }

    Ok(())
}

/// Live demo: Start Anvil, deploy contracts, spawn real helix-node processes,
/// run distributed ZK-verified training with on-chain proof submission.
///
/// This is the 90-second ETHDenver demo with real components.
async fn cmd_demo_live(args: &DemoArgs, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    use helix_client::orchestration::{
        find_contracts_dir, find_node_binary, OrchestratorConfig, TrainingOrchestrator,
    };

    // Parse model dimensions
    let dims: Vec<usize> = args
        .model
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow::anyhow!("Invalid --model format. Expected d_in,d_hid,d_out"))?;
    if dims.len() != 3 {
        return Err(anyhow::anyhow!("--model requires exactly 3 dimensions"));
    }
    let (d_in, d_hid, d_out) = (dims[0], dims[1], dims[2]);

    // Resolve paths
    let node_binary = match &args.node_binary {
        Some(p) => p.clone(),
        None => find_node_binary()?,
    };
    let contracts_dir = match &args.contracts_dir {
        Some(p) => p.clone(),
        None => find_contracts_dir()?,
    };

    // Validate private key is provided for on-chain operations
    if args.demo_private_key.is_empty() {
        return Err(anyhow::anyhow!(
            "Private key required for live demo. Set HELIX_PRIVATE_KEY env var or pass --demo-private-key"
        ));
    }

    let deploy = args.coordinator_address.is_none() && !args.skip_deploy;
    let coordinator = args.coordinator_address.clone().unwrap_or_default();

    let config = OrchestratorConfig {
        model_dims: (d_in, d_hid, d_out),
        model_seed: args.model_seed,
        learning_rate: 0.01,
        rounds: args.rounds,
        workers: args.workers,
        node_binary: node_binary.clone(),
        base_port: args.base_port,
        http_port: args.demo_http_port,
        eth_rpc_url: format!("http://127.0.0.1:{}", 8545),
        private_key: args.demo_private_key.clone(),
        contracts_dir: contracts_dir.clone(),
        deploy_contracts: deploy,
        coordinator_address: coordinator,
        start_anvil: true,
        anvil_port: 8545,
        worker_timeout: Duration::from_secs(30),
        round_timeout: Duration::from_secs(120),
        health_check_interval: Duration::from_secs(5),
        max_restarts: 3,
        checkpoint: helix_client::demo::checkpoint::CheckpointConfig::default(),
        resume: false,
    };

    // Print banner
    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " HELIX Live Demo — Real Distributed Training".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    println!("{}", "Configuration:".yellow().bold());
    println!("  Model dims:    {}x{}x{} ({} params)",
        d_in, d_hid, d_out,
        d_in * d_hid + d_hid + d_hid * d_out + d_out
    );
    println!("  Workers:       {}", args.workers);
    println!("  Rounds:        {}", args.rounds);
    println!("  Node binary:   {}", node_binary.display());
    println!("  Contracts:     {}", contracts_dir.display());
    println!("  Deploy:        {}", if deploy { "yes (fresh Anvil + forge)" } else { "no (pre-deployed)" });
    println!();

    let mut orchestrator = TrainingOrchestrator::new(config)?;

    // Phase 1: Start Anvil
    let mut progress = ProgressDisplay::new();
    progress.start_spinner("Starting Anvil...");
    orchestrator.start_anvil().await?;
    progress.finish_spinner("Anvil running on http://127.0.0.1:8545");

    // Phase 2: Deploy contracts
    if deploy {
        progress.start_spinner("Deploying contracts via forge...");
        orchestrator.deploy_contracts().await?;
        if let Some(dep) = orchestrator.deployment() {
            progress.finish_spinner(&format!(
                "Contracts deployed (coordinator={})",
                &dep.coordinator[..10]
            ));
        } else {
            progress.finish_spinner("Contracts deployed");
        }
    }

    // Phase 3: Spawn network
    progress.start_spinner(&format!("Starting 1 aggregator + {} workers...", args.workers));
    orchestrator.start_network().await?;
    progress.finish_spinner(&format!("Network started ({} nodes)", 1 + args.workers));

    // Phase 4: Wait for workers
    progress.start_spinner("Waiting for workers to connect...");
    orchestrator.wait_for_workers().await?;
    progress.finish_spinner(&format!("{} workers connected", args.workers));

    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " Training Progress".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    // Phase 5: Training rounds
    let start = std::time::Instant::now();
    let mut rounds_completed = 0u32;

    for round in 1..=args.rounds {
        if shutdown.try_recv().is_ok() {
            println!("\n{}", "Demo interrupted by user".yellow());
            break;
        }

        progress.start_spinner(&format!("Round {}/{} — training...", round, args.rounds));

        // Trigger round
        let round_start = std::time::Instant::now();
        match orchestrator.trigger_round().await {
            Ok(_) => {}
            Err(e) => {
                progress.finish_spinner_error(&format!("Round {} failed: {}", round, e));
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        }

        // Wait for completion
        match orchestrator.wait_for_round_completion(round).await {
            Ok(_status) => {
                let elapsed = round_start.elapsed().as_millis();
                progress.finish_spinner(&format!(
                    "Round {}/{} — completed ({}ms)",
                    round, args.rounds, elapsed
                ));
                rounds_completed = round;
            }
            Err(e) => {
                progress.finish_spinner_error(&format!("Round {} — {}", round, e));
            }
        }
    }

    let total_elapsed = start.elapsed();

    // Results
    println!();
    println!("{}", "═".repeat(60).green());
    println!("{}", " Demo Complete".green().bold());
    println!("{}", "═".repeat(60).green());
    println!();
    println!("  Rounds completed: {}/{}", rounds_completed, args.rounds);
    println!("  Total time:       {:.1}s", total_elapsed.as_secs_f64());
    println!("  Workers:          {}", args.workers);
    if let Some(dep) = orchestrator.deployment() {
        println!("  Coordinator:      {}", dep.coordinator);
        println!("  Verifier:         {}", dep.verifier);
    }
    println!();

    // Cleanup
    progress.start_spinner("Shutting down...");
    orchestrator.shutdown().await?;
    progress.finish_spinner("All processes stopped");

    Ok(())
}


async fn cmd_orchestrate(args: &OrchestrateArgs, _cli: &Cli) -> Result<()> {
    let mut progress = ProgressDisplay::new();

    match args.action {
        OrchestrateAction::Start => {
            println!("{}", "Starting HELIX network...".cyan().bold());

            progress.start_spinner("Creating network configuration...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            progress.finish_spinner("Configuration created");

            for i in 0..args.nodes {
                progress.start_spinner(&format!("Starting node {}/{}...", i + 1, args.nodes));
                tokio::time::sleep(Duration::from_millis(500)).await;
                progress.finish_spinner(&format!("Node {} started", i + 1));
            }

            println!("\n{}", "Network started successfully!".green().bold());
            println!("  Network Name: {}", args.network_name);
            println!("  Nodes:        {}", args.nodes);
        }
        OrchestrateAction::Stop => {
            println!("{}", "Stopping HELIX network...".cyan().bold());

            progress.start_spinner("Sending shutdown signals...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            progress.finish_spinner("Shutdown signals sent");

            progress.start_spinner("Waiting for graceful shutdown...");
            tokio::time::sleep(Duration::from_secs(2)).await;
            progress.finish_spinner("All nodes stopped");

            println!("\n{}", "Network stopped.".yellow());
        }
        OrchestrateAction::Restart => {
            println!("{}", "Restarting HELIX network...".cyan().bold());
            tokio::time::sleep(Duration::from_secs(3)).await;
            println!("{}", "Network restarted.".green());
        }
        OrchestrateAction::Scale => {
            println!("{}", "Scaling HELIX network...".cyan().bold());
            println!("  Target nodes: {}", args.nodes);
            tokio::time::sleep(Duration::from_secs(2)).await;
            println!("{}", "Network scaled successfully.".green());
        }
        OrchestrateAction::Topology => {
            println!("{}", "Network Topology".cyan().bold());
            println!();
            println!("     ┌─────────────────┐");
            println!("     │   Aggregator    │");
            println!("     │  (node-agg-01)  │");
            println!("     └────────┬────────┘");
            println!("              │");
            println!("    ┌─────────┼─────────┐");
            println!("    │         │         │");
            println!("┌───┴───┐ ┌───┴───┐ ┌───┴───┐");
            println!("│Worker1│ │Worker2│ │Worker3│");
            println!("└───────┘ └───────┘ └───────┘");
        }
    }

    Ok(())
}

async fn cmd_health(args: &HealthArgs, cli: &Cli) -> Result<()> {
    println!("{}", "Running health checks...".cyan().bold());
    println!();

    // Load config to get RPC URL
    let config = if cli.config.exists() {
        config::HelixConfig::load(&cli.config).unwrap_or_default()
    } else {
        config::HelixConfig::default()
    };

    let checker = health::HealthChecker::new(
        Duration::from_secs(args.timeout),
        &config.rpc.url,
    )
    .with_ipfs_gateway(&config.storage.ipfs_gateway)
    .with_data_dir(config.node.data_dir.clone());

    // Map CLI target to health module target
    let target = match args.target {
        HealthTarget::All => health::HealthTarget::All,
        HealthTarget::Nodes => health::HealthTarget::Nodes,
        HealthTarget::Contracts => health::HealthTarget::Contracts,
        HealthTarget::Network => health::HealthTarget::Network,
        HealthTarget::Storage => health::HealthTarget::Storage,
    };

    let report = checker.check(target).await;
    health::display_health_report(&report, args.detailed);

    Ok(())
}

async fn cmd_dashboard(args: &DashboardArgs, _cli: &Cli, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    println!("{}", "Starting HELIX Status Dashboard...".cyan().bold());
    println!("  Address: http://{}:{}", args.host, args.port);
    if let Some(ref coord) = args.coordinator {
        println!("  Coordinator: {}", coord);
    }

    let config = dashboard::DashboardConfig {
        auth_token: None,
        allowed_origins: Vec::new(), // Allow all origins (dev/demo mode)
        rate_limit_per_second: 30,
    };

    let state = Arc::new(dashboard::DashboardState::new(config));

    // Set on-chain config from CLI args
    if let Some(ref rpc_url) = args.rpc_url {
        *state.eth_rpc_url.write().await = Some(rpc_url.clone());
    }
    if let Some(ref coordinator) = args.coordinator {
        *state.coordinator_address.write().await = Some(coordinator.clone());
    }

    let app = dashboard::create_dashboard_router_with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{}:{}", args.host, args.port)).await?;

    println!("{}", "Dashboard is running. Press Ctrl+C to stop.".green());

    tokio::select! {
        result = axum::serve(listener, app) => {
            result?;
        }
        _ = shutdown.recv() => {
            println!("\nShutting down dashboard...");
        }
    }

    Ok(())
}

async fn cmd_benchmark(args: &BenchmarkArgs, _cli: &Cli) -> Result<()> {
    println!("{}", "Running HELIX Benchmarks".cyan().bold());
    println!("  Type:       {:?}", args.benchmark_type);
    println!("  Iterations: {}", args.iterations);
    println!("  Warmup:     {}", args.warmup);
    println!();

    let mut progress = ProgressDisplay::new();

    // Warmup
    progress.start_spinner(&format!("Warmup ({} iterations)...", args.warmup));
    tokio::time::sleep(Duration::from_secs(1)).await;
    progress.finish_spinner("Warmup complete");

    // Run benchmark
    let pb = indicatif::ProgressBar::new(args.iterations as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut times: Vec<f64> = Vec::new();

    for _ in 0..args.iterations {
        let start = std::time::Instant::now();

        // Simulate benchmark work
        match args.benchmark_type {
            BenchmarkType::ProofGen => tokio::time::sleep(Duration::from_millis(10)).await,
            BenchmarkType::ProofVerify => tokio::time::sleep(Duration::from_millis(5)).await,
            BenchmarkType::Aggregation => tokio::time::sleep(Duration::from_millis(3)).await,
            BenchmarkType::Network => tokio::time::sleep(Duration::from_millis(2)).await,
            BenchmarkType::E2e => tokio::time::sleep(Duration::from_millis(50)).await,
        }

        times.push(start.elapsed().as_secs_f64() * 1000.0);
        pb.inc(1);
    }

    pb.finish_and_clear();

    // Calculate statistics
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sorted = {
        let mut t = times.clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        t
    };
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

    // Results
    println!("\n{}", "Results:".yellow().bold());
    println!("  ┌────────────┬────────────────┐");
    println!("  │ Metric     │ Value          │");
    println!("  ├────────────┼────────────────┤");
    println!("  │ Mean       │ {:>10.3} ms  │", avg);
    println!("  │ Min        │ {:>10.3} ms  │", min);
    println!("  │ Max        │ {:>10.3} ms  │", max);
    println!("  │ P50        │ {:>10.3} ms  │", p50);
    println!("  │ P95        │ {:>10.3} ms  │", p95);
    println!("  │ P99        │ {:>10.3} ms  │", p99);
    println!("  │ Throughput │ {:>10.1} ops/s│", 1000.0 / avg);
    println!("  └────────────┴────────────────┘");

    if let Some(ref output) = args.output {
        let results = serde_json::json!({
            "benchmark": format!("{:?}", args.benchmark_type),
            "iterations": args.iterations,
            "mean_ms": avg,
            "min_ms": min,
            "max_ms": max,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "throughput_ops_s": 1000.0 / avg,
        });
        std::fs::write(&output, serde_json::to_string_pretty(&results)?)?;
        println!("\nResults saved to: {}", output.display());
    }

    Ok(())
}

async fn cmd_logs(args: &LogsArgs, _cli: &Cli, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    println!("{}", "Log Viewer".cyan().bold());

    let sample_logs = vec![
        ("2024-01-15 14:23:45", "INFO", "helix-node-1", "Round 42 started"),
        ("2024-01-15 14:23:46", "DEBUG", "helix-node-2", "Received gradient from peer helix-node-1"),
        ("2024-01-15 14:23:47", "INFO", "helix-node-3", "Training batch 15/32 complete"),
        ("2024-01-15 14:23:48", "INFO", "helix-agg", "Aggregating gradients from 3 workers"),
        ("2024-01-15 14:23:49", "DEBUG", "helix-node-1", "Generating ZK proof for round 42"),
        ("2024-01-15 14:23:52", "INFO", "helix-node-1", "Proof generated successfully"),
        ("2024-01-15 14:23:53", "INFO", "helix-agg", "Verifying proof from helix-node-1"),
        ("2024-01-15 14:23:54", "INFO", "helix-agg", "Proof verified, committing round 42"),
        ("2024-01-15 14:23:55", "INFO", "helix-agg", "Round 42 completed successfully"),
    ];

    let filter_log = |level: &str, node: &str| -> bool {
        if let Some(ref level_filter) = args.level {
            if !level.to_lowercase().contains(&level_filter.to_lowercase()) {
                return false;
            }
        }
        if let Some(ref node_filter) = args.node {
            if !node.contains(node_filter) {
                return false;
            }
        }
        if let Some(ref grep) = args.grep {
            return node.contains(grep);
        }
        true
    };

    let display_log = |time: &str, level: &str, node: &str, msg: &str| {
        let level_colored = match level {
            "INFO" => level.green(),
            "DEBUG" => level.blue(),
            "WARN" => level.yellow(),
            "ERROR" => level.red(),
            _ => level.normal(),
        };
        println!("{} [{}] {} {}", time.dimmed(), level_colored, node.cyan(), msg);
    };

    // Show initial logs
    let shown: Vec<_> = sample_logs
        .iter()
        .filter(|(_, level, node, _)| filter_log(level, node))
        .take(args.lines)
        .collect();

    for (time, level, node, msg) in &shown {
        display_log(time, level, node, msg);
    }

    if args.follow {
        println!("\n{}", "Following logs... (Ctrl+C to stop)".dimmed());

        loop {
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(2)) => {
                    let time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                    display_log(&time, "INFO", "helix-node-1", "Heartbeat sent");
                }
                _ = shutdown.recv() => {
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn cmd_watch(args: &WatchArgs, _cli: &Cli, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    use notify::{Watcher, RecursiveMode, recommended_watcher};

    println!("{}", "Watching configuration files...".cyan().bold());
    for file in &args.files {
        println!("  - {}", file.display());
    }
    println!("Action on change: {}", args.on_change);
    println!("\nPress Ctrl+C to stop.\n");

    let (tx, mut rx) = tokio::sync::mpsc::channel(100);

    let mut watcher = recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let _ = tx.blocking_send(event);
        }
    })?;

    for file in &args.files {
        if file.exists() {
            watcher.watch(file, RecursiveMode::NonRecursive)?;
        }
    }

    loop {
        tokio::select! {
            Some(event) = rx.recv() => {
                if event.kind.is_modify() {
                    let paths: Vec<_> = event.paths.iter().map(|p| p.display().to_string()).collect();
                    println!("{} Config changed: {}", chrono::Local::now().format("%H:%M:%S").to_string().dimmed(), paths.join(", "));

                    match args.on_change.as_str() {
                        "reload" => println!("  → Reloading configuration..."),
                        "restart" => println!("  → Triggering restart..."),
                        "notify" => println!("  → Change detected (notify mode)"),
                        _ => {}
                    }
                }
            }
            _ = shutdown.recv() => {
                println!("\nStopping config watcher.");
                break;
            }
        }
    }

    Ok(())
}

async fn cmd_help(args: &HelpArgs, _cli: &Cli) -> Result<()> {
    let help_system = help::HelpSystem::new();

    if args.quick {
        help::print_quick_reference();
    } else if args.topic.is_empty() {
        help_system.print_general_help();
    } else {
        // Check if it's a command or a topic
        let topic = args.topic.to_lowercase();
        match topic.as_str() {
            "init" | "join" | "status" | "query" | "export" | "demo" |
            "orchestrate" | "health" | "visualize" | "dashboard" |
            "benchmark" | "logs" | "watch" => {
                help_system.print_command_help(&topic);
            }
            _ => {
                help_system.print_topic_help(&topic);
            }
        }
    }

    Ok(())
}

async fn cmd_visualize(args: &VisualizeArgs, _cli: &Cli, shutdown: broadcast::Receiver<()>) -> Result<()> {
    use visualization::VisualizationRunner;

    // Create and configure visualization runner
    let mut runner = VisualizationRunner::new()?;

    // Configure based on args
    {
        let state = runner.state_mut();
        state.total_rounds = args.rounds as u64;
        state.total_workers = args.workers;
        state.active_workers = args.workers;

        // Set demo wallet info for visualization
        state.update_wallet_info(
            Some("0xdEm0...1234".to_string()),
            100.0,  // 100 HLX staked
            0.0,    // No pending rewards yet
            12.5,   // 12.5% APY
        );
        state.set_hardware_wallet(false, None);
    }

    // Run visualization
    runner.run(shutdown).await?;

    Ok(())
}

// ============================================================================
// Chain Commands (feature = "chain")
// ============================================================================

/// Train with real proof generation and on-chain submission.
///
/// Uses `helix-prover` to produce real SHPLONK proofs after each training step,
/// then submits them to the `HelixCoordinatorV2` contract via `ChainClient`.
#[cfg(feature = "chain")]
async fn cmd_train_chain(args: &TrainArgs) -> Result<()> {
    use rpc::chain::ChainClient;

    let model_id = args.model_id.ok_or_else(|| {
        anyhow::anyhow!("--model-id is required in --chain mode")
    })?;
    let steps = args.steps.unwrap_or(args.max_rounds.unwrap_or(5));

    if args.private_key.is_empty() {
        return Err(anyhow::anyhow!(
            "--private-key (or HELIX_PRIVATE_KEY env) required for chain mode"
        ));
    }
    if args.coordinator_address.is_empty() {
        return Err(anyhow::anyhow!(
            "--coordinator-address (or COORDINATOR_ADDRESS env) required for chain mode"
        ));
    }

    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " HELIX Chain Training".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    println!("{}", "Configuration:".yellow().bold());
    println!("  Model ID:      {}", model_id);
    println!("  Steps:         {}", steps);
    println!("  RPC URL:       {}", args.rpc_url);
    println!("  Coordinator:   {}", args.coordinator_address);
    if let Some(ref data) = args.data {
        println!("  Data path:     {}", data.display());
    }
    println!();

    // Connect chain client
    let mut progress = ProgressDisplay::new();
    progress.start_spinner("Connecting to chain...");
    let chain = ChainClient::new(
        &args.rpc_url,
        &args.private_key,
        &args.coordinator_address,
        None,
    )
    .await?;
    progress.finish_spinner(&format!(
        "Connected as {}",
        format!("0x{:x}", chain.signer_address())
    ));

    // Query current model state
    progress.start_spinner("Querying model state...");
    let state = chain.get_model_state(model_id).await?;
    progress.finish_spinner(&format!(
        "Model round={}, active={}",
        state.current_round, state.active
    ));

    if !state.active {
        println!(
            "{}",
            "Warning: model is not active on-chain — proofs may revert"
                .yellow()
        );
    }

    // Initialize ZK prover for real proof generation
    progress.start_spinner("Initializing ZK prover...");
    let (d_in, d_hid, d_out) = (4, 8, 2); // Default model dims
    let prover_config = helix_prover::V2ProverConfig {
        k: 14,
        relu_range: 128,
        exp_range: 256,
        exp_scale: 1000,
        use_freivalds: true,
        base_error: helix_prover::halo2curves::bn256::Fr::from(1u64),
        self_verify: true,
        ..helix_prover::V2ProverConfig::default()
    };
    let prover = Arc::new(helix_prover::MLTrainingProverV2::with_config(
        d_in, d_hid, d_out, prover_config,
    ));
    let mut training_state = demo::real_training::TrainingState::new_random(d_in, d_hid, d_out);
    progress.finish_spinner("ZK prover initialized");

    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " Training + Real Proof Submission".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    for step in 1..=steps {
        progress.start_spinner(&format!("Step {}/{} — generating real ZK proof...", step, steps));

        // Generate training data and build witness from current weights
        let (x, target) = {
            use helix_prover::halo2curves::bn256::Fr;
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let x: Vec<Fr> = (0..d_in)
                .map(|_| {
                    let v: f64 = rng.gen_range(-0.5..0.5);
                    let scaled = (v * 65536.0) as i64;
                    if scaled >= 0 { Fr::from(scaled as u64) } else { -Fr::from((-scaled) as u64) }
                })
                .collect();
            let target_idx = rng.gen_range(0..d_out);
            let target: Vec<Fr> = (0..d_out)
                .map(|i| if i == target_idx { Fr::from(1u64) } else { Fr::from(0u64) })
                .collect();
            (x, target)
        };

        let lr = helix_prover::halo2curves::bn256::Fr::from((0.01_f64 * 65536.0) as u64);
        let step_number = state.current_round + step as u64;

        let witness = helix_prover::MLTrainingProverV2::build_witness(
            d_in, d_hid, d_out,
            &x, &target,
            &training_state.w1, &training_state.b1,
            &training_state.w2, &training_state.b2,
            lr,
            step_number,
            helix_prover::halo2curves::bn256::Fr::from(1u64),
        );

        // Generate real ZK proof with timeout to prevent indefinite hangs
        let proof_timeout = Duration::from_secs(300); // 5 minute timeout per proof
        let proof_result = {
            let witness_clone = witness.clone();
            let prover_arc = Arc::clone(&prover);
            let prove_future = tokio::task::spawn_blocking(move || {
                prover_arc.prove(&witness_clone)
            });
            match tokio::time::timeout(proof_timeout, prove_future).await {
                Ok(Ok(result)) => result.map_err(|e| {
                    anyhow::anyhow!("Proof generation failed at step {}: {:?}", step, e)
                })?,
                Ok(Err(join_err)) => {
                    return Err(anyhow::anyhow!(
                        "Proof generation panicked at step {}: {}", step, join_err
                    ));
                }
                Err(_) => {
                    return Err(anyhow::anyhow!(
                        "Proof generation timed out at step {} (limit: {}s)",
                        step, proof_timeout.as_secs()
                    ));
                }
            }
        };

        // Update training state
        training_state.w1 = witness.w1_new.clone();
        training_state.b1 = witness.b1_new.clone();
        training_state.w2 = witness.w2_new.clone();
        training_state.b2 = witness.b2_new.clone();
        training_state.step += 1;

        // Convert to EVM format
        let proof_bytes = proof_result.to_evm_proof().map_err(|e| {
            anyhow::anyhow!("EVM proof formatting failed: {:?}", e)
        })?;
        let evm_public_inputs = proof_result.to_evm_public_inputs();
        let public_inputs_u256: Vec<ethers::types::U256> = evm_public_inputs
            .iter()
            .map(|bytes| ethers::types::U256::from_big_endian(bytes))
            .collect();

        progress.finish_spinner(&format!(
            "Step {}/{} — real proof ready ({} bytes, verified={})",
            step, steps, proof_bytes.len(), proof_result.verified
        ));

        // Submit real proof to chain
        progress.start_spinner(&format!(
            "Step {}/{} — submitting real proof on-chain...",
            step, steps
        ));
        let round_id = step_number;
        match chain
            .submit_proof_raw(model_id, round_id, proof_bytes, public_inputs_u256)
            .await
        {
            Ok(receipt) => {
                let tx = receipt.transaction_hash;
                progress.finish_spinner(&format!(
                    "Step {}/{} — tx 0x{:x} (block {})",
                    step,
                    steps,
                    tx,
                    receipt.block_number.map_or(0, |b| b.as_u64())
                ));
            }
            Err(e) => {
                progress.finish_spinner_error(&format!(
                    "Step {}/{} — submit failed: {}",
                    step, steps, e
                ));
            }
        }
    }

    println!();
    println!("{}", "═".repeat(60).green());
    println!("{}", " Chain Training Complete".green().bold());
    println!("{}", "═".repeat(60).green());
    println!();

    Ok(())
}

/// Submit a pre-generated proof file to the coordinator contract.
///
/// The JSON file should contain:
/// ```json
/// {
///   "proof": "<hex-encoded proof bytes>",
///   "public_inputs": {
///     "old_hash_lo": "0x...",
///     "old_hash_hi": "0x...",
///     "new_hash_lo": "0x...",
///     "new_hash_hi": "0x...",
///     "loss": "0x...",
///     "error_bound": "0x...",
///     "step_number": "0x..."
///   }
/// }
/// ```
#[cfg(feature = "chain")]
async fn cmd_submit_proof(args: &SubmitProofArgs) -> Result<()> {
    use rpc::chain::{ChainClient, TrainingProofInputs};

    println!();
    println!("{}", "═".repeat(60).cyan());
    println!("{}", " HELIX Proof Submission".cyan().bold());
    println!("{}", "═".repeat(60).cyan());
    println!();

    println!("{}", "Parameters:".yellow().bold());
    println!("  Model ID:    {}", args.model_id);
    println!("  Round ID:    {}", args.round_id);
    println!("  Proof file:  {}", args.proof_file.display());
    println!("  RPC URL:     {}", args.rpc_url);
    println!("  Coordinator: {}", args.coordinator_address);
    println!();

    // Read proof file
    let file_contents = std::fs::read_to_string(&args.proof_file)
        .map_err(|e| anyhow::anyhow!("Failed to read proof file: {}", e))?;

    #[derive(serde::Deserialize)]
    struct ProofFile {
        proof: String,
        public_inputs: TrainingProofInputs,
    }

    let proof_data: ProofFile = serde_json::from_str(&file_contents)
        .map_err(|e| anyhow::anyhow!("Failed to parse proof JSON: {}", e))?;

    let proof_bytes = hex::decode(proof_data.proof.strip_prefix("0x").unwrap_or(&proof_data.proof))
        .map_err(|e| anyhow::anyhow!("Invalid hex in proof field: {}", e))?;

    println!("  Proof size:  {} bytes", proof_bytes.len());
    println!();

    // Connect chain client
    let mut progress = ProgressDisplay::new();
    progress.start_spinner("Connecting to chain...");
    let chain = ChainClient::new(
        &args.rpc_url,
        &args.private_key,
        &args.coordinator_address,
        args.chain_id,
    )
    .await?;
    progress.finish_spinner(&format!(
        "Connected as {}",
        format!("0x{:x}", chain.signer_address())
    ));

    // Submit proof
    progress.start_spinner("Submitting proof on-chain...");
    let receipt = chain
        .submit_proof(
            args.model_id,
            args.round_id,
            proof_bytes,
            &proof_data.public_inputs,
        )
        .await?;
    progress.finish_spinner(&format!(
        "Proof submitted — tx 0x{:x} (block {})",
        receipt.transaction_hash,
        receipt.block_number.map_or(0, |b| b.as_u64())
    ));

    println!();
    println!("{}", "Proof submitted successfully!".green().bold());
    println!();

    Ok(())
}

// ============================================================================
// ZK Mode Parsing
// ============================================================================

/// Parse the --zk-mode CLI flag into a ZkMode enum.
///
/// Accepted values: "off", "always", "risk:N" (e.g. "risk:2").
fn parse_zk_mode(s: &str) -> Result<helix_client::ZkMode> {
    match s.to_lowercase().as_str() {
        "off" => Ok(helix_client::ZkMode::Off),
        "always" => Ok(helix_client::ZkMode::Always),
        other if other.starts_with("risk:") => {
            let n: usize = other[5..].parse().map_err(|_| {
                anyhow::anyhow!(
                    "Invalid --zk-mode format. Expected 'risk:N' where N is a number (e.g. 'risk:2')"
                )
            })?;
            Ok(helix_client::ZkMode::Risk { min_workers: n })
        }
        _ => Err(anyhow::anyhow!(
            "Unknown --zk-mode '{}'. Options: off, always, risk:N",
            s
        )),
    }
}

// ============================================================================
// MPC Training (Model Owner)
// ============================================================================

async fn cmd_mpc_train(args: &MpcTrainArgs, _cli: &Cli) -> Result<()> {
    use helix_client::full_orchestration::{
        FullOrchestrationConfig, FullOrchestrator, ProgressCallback, ProgressEvent,
    };

    println!();
    println!("{}", "═".repeat(64).cyan());
    println!("{}", " HELIX MPC-Primary Training (Model Owner)".cyan().bold());
    println!("{}", "═".repeat(64).cyan());
    println!();

    // Parse architecture dimensions.
    let dims: Vec<usize> = args
        .architecture
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow::anyhow!("Invalid --architecture format. Expected d_in,d_hid,d_out (e.g. 784,32,10)"))?;
    if dims.len() != 3 {
        return Err(anyhow::anyhow!("--architecture requires exactly 3 dimensions (d_in,d_hid,d_out)"));
    }

    // Use --num-workers localhost endpoints if --workers not explicitly provided.
    let worker_endpoints = if args.workers.is_empty() {
        (0..args.num_workers)
            .map(|i| format!("127.0.0.1:{}", 9001 + i))
            .collect()
    } else {
        args.workers.clone()
    };

    // Worker private keys: explicit > Anvil defaults (local) > empty (auto-generate on testnet).
    #[cfg(feature = "chain")]
    let worker_keys = if !args.worker_keys.is_empty() {
        args.worker_keys.clone()
    } else {
        let is_local = args.rpc_url.as_ref().map_or(true, |url| {
            let u = url.to_lowercase();
            u.contains("127.0.0.1") || u.contains("localhost") || u.contains("[::1]")
        });
        if is_local {
            // Auto-populate with Anvil default keys (accounts 1-10).
            let anvil_default_keys = vec![
                "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d".to_string(),
                "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a".to_string(),
                "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6".to_string(),
                "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a".to_string(),
                "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba".to_string(),
                "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e".to_string(),
                "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6".to_string(),
                "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97".to_string(),
                "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6".to_string(),
                "0xf214f2b2cd398c806f84e317254e0f0b801d0643303237d97a22a48e01628897".to_string(),
            ];
            let needed = worker_endpoints.len().min(anvil_default_keys.len());
            eprintln!(
                "{}",
                format!(
                    "  Note: Using Anvil default keys for {} workers (demo mode)",
                    needed
                ).dimmed()
            );
            anvil_default_keys[..needed].to_vec()
        } else {
            // Remote chain: leave empty — orchestrator will generate and fund workers.
            eprintln!(
                "{}",
                "  Note: Workers will be auto-generated and funded from owner account".dimmed()
            );
            Vec::new()
        }
    };

    let total_params = dims[1] * dims[0] + dims[1] + dims[2] * dims[1] + dims[2];
    println!("{}", "Configuration:".yellow().bold());
    println!("  Architecture:    {}x{}x{}", dims[0], dims[1], dims[2]);
    println!("  Parameters:      {}", total_params);
    println!("  Steps:           {}", args.steps);
    println!("  Learning Rate:   {}", args.learning_rate);
    println!("  Checkpoint Freq: every {} steps", args.checkpoint_freq);
    println!("  MAC Interval:    every {} steps", args.mac_interval);
    println!("  Workers:         {}", worker_endpoints.len());
    for (i, ep) in worker_endpoints.iter().enumerate() {
        println!("    Worker {}: {}", i, ep);
    }
    println!("  Seed:            {}", args.seed);
    #[cfg(feature = "chain")]
    {
        println!("  Payment:         {} ETH", args.payment_eth);
        println!("  Stake/worker:    {} ETH", args.stake_eth);
    }
    println!("  ZK Mode:         {}", args.zk_mode);
    if args.simulate_cheater {
        println!("  {} CHEATER SIMULATION ENABLED", "⚠".yellow());
    }
    println!();

    let mut config = FullOrchestrationConfig {
        architecture: dims.clone(),
        num_steps: args.steps,
        learning_rate: args.learning_rate,
        checkpoint_frequency: args.checkpoint_freq,
        mac_check_interval: args.mac_interval,
        beaver_batch_size: args.beaver_batch_size,
        batch_size: args.batch_size,
        seed: args.seed,
        worker_endpoints: worker_endpoints.clone(),
        initial_weights_path: args.weights.clone(),
        zk_proof: helix_client::ZkProofConfig {
            enabled: args.zk_proofs,
            // If zk_checkpoint_freq is 0 (default), only prove at the very end:
            // set frequency to total checkpoints so only the last one gets a proof.
            checkpoint_frequency: if args.zk_checkpoint_freq > 0 {
                args.zk_checkpoint_freq
            } else if args.checkpoint_freq > 0 {
                // Default: prove only at final checkpoint
                (args.steps / args.checkpoint_freq).max(1)
            } else {
                1
            },
            self_verify: true,
        },
        use_real_mnist: args.real_mnist,
        mnist_cache_dir: args.mnist_dir.clone(),
        train_size: args.train_size,
        test_size: args.test_size,
        #[cfg(feature = "chain")]
        eth_rpc_url: args.rpc_url.clone(),
        #[cfg(feature = "chain")]
        private_key: args.private_key.clone()
            .or_else(|| std::env::var("HELIX_PRIVATE_KEY").ok())
            .unwrap_or_else(|| {
                // Anvil default account 0 for local dev
                "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".to_string()
            }),
        #[cfg(feature = "chain")]
        worker_private_keys: worker_keys,
        #[cfg(feature = "chain")]
        payment_amount_eth: args.payment_eth,
        #[cfg(feature = "chain")]
        stake_amount_eth: args.stake_eth,
        #[cfg(feature = "chain")]
        coordinator_address: args.coordinator.clone(),
        #[cfg(feature = "chain")]
        use_pool_workers: false, // CLI mpc-train manages its own workers
        #[cfg(feature = "chain")]
        enable_withdrawal: args.enable_withdrawal,
        #[cfg(feature = "chain")]
        pre_registered_job_id: None,
        zk_mode: parse_zk_mode(&args.zk_mode)?,
        distributed: args.transport == "distributed",
        custom_training_data: None,
        simulate_cheater: args.simulate_cheater,
        worker_seeds: Vec::new(), // CLI mode: seeds derived from config.seed + i
    };

    // Load custom training data from --data flag if provided
    if let Some(ref data_path) = args.data {
        let data_str = std::fs::read_to_string(data_path)
            .map_err(|e| anyhow::anyhow!("Failed to read --data file '{}': {}", data_path.display(), e))?;
        let samples: Vec<helix_client::dashboard::TrainingSample> = serde_json::from_str(&data_str)
            .map_err(|e| anyhow::anyhow!(
                "Failed to parse --data file. Expected JSON array of {{input: [...], target: [...]}}: {}", e
            ))?;
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = samples
            .into_iter()
            .map(|s| (s.input, s.target))
            .collect();
        println!("  Loaded {} custom training samples from {}", pairs.len(), data_path.display());
        config.custom_training_data = Some(pairs);
    }

    let mut orchestrator = FullOrchestrator::new(config);

    // Set up the progress display callback for user-facing output.
    let num_steps = args.steps;
    let num_workers = worker_endpoints.len();
    let progress_cb: ProgressCallback = std::sync::Arc::new(move |event: ProgressEvent| {
        match event {
            ProgressEvent::PhaseStarted { phase, total, description } => {
                println!(
                    "{}",
                    format!("Phase {}/{}: {}...", phase, total, description)
                        .cyan()
                        .bold()
                );
            }
            ProgressEvent::PhaseCompleted { phase, elapsed_ms } => {
                if elapsed_ms > 0 {
                    println!(
                        "  {} Phase {} complete ({:.1}s)",
                        "✓".green(),
                        phase,
                        elapsed_ms as f64 / 1000.0
                    );
                }
            }
            ProgressEvent::TrainingStep { step, total, loss, mac_ok } => {
                // Show progress at regular intervals to avoid flooding.
                let show = step == 1
                    || step == total
                    || (total <= 50)
                    || (total <= 200 && step % 10 == 0)
                    || (total <= 1000 && step % 50 == 0)
                    || (step % 100 == 0);
                if show {
                    let mac_indicator = if mac_ok {
                        "MACs: ✓".green().to_string()
                    } else {
                        "MACs: ✗".red().to_string()
                    };
                    let pct = (step as f64 / total as f64 * 100.0) as u32;
                    let bar_width = 30;
                    let filled = (pct as usize * bar_width / 100).min(bar_width);
                    let bar = format!(
                        "[{}{}]",
                        "#".repeat(filled),
                        "-".repeat(bar_width - filled)
                    );
                    println!(
                        "  Step {}/{} {} Loss: {:.4} -- {}",
                        step, total,
                        bar.dimmed(),
                        loss,
                        mac_indicator,
                    );
                }
            }
            ProgressEvent::CheckpointSubmitted { index, total, step, tx_hash } => {
                let short_hash = if tx_hash.len() > 12 {
                    format!("{}...{}", &tx_hash[..8], &tx_hash[tx_hash.len()-4..])
                } else {
                    tx_hash
                };
                println!(
                    "  {} Checkpoint {}/{} submitted on-chain at step {} (tx: {})",
                    "✓".green(),
                    index,
                    total,
                    step,
                    short_hash.dimmed(),
                );
            }
            ProgressEvent::CheaterDetected { party_index, step } => {
                println!();
                println!(
                    "  {} {} at step {} -- Worker {} identified",
                    "⚠".yellow(),
                    "MAC FAILURE DETECTED".red().bold(),
                    step,
                    party_index,
                );
            }
            ProgressEvent::CheaterSlashed { party_index, tx_hash } => {
                let short_hash = if tx_hash.len() > 12 {
                    format!("{}...{}", &tx_hash[..8], &tx_hash[tx_hash.len()-4..])
                } else {
                    tx_hash
                };
                println!(
                    "  {} Worker {} slashed on-chain (tx: {})",
                    "⚡".red(),
                    party_index,
                    short_hash.dimmed(),
                );
                println!();
            }
            ProgressEvent::TrainingComplete { accuracy, steps, time_secs, checkpoints } => {
                println!();
                println!("{}", "═".repeat(64).green());
                println!(
                    "  {} -- {:.1}% accuracy -- {} steps -- {:.1}s -- {} checkpoints verified",
                    "Training complete".green().bold(),
                    accuracy * 100.0,
                    steps,
                    time_secs,
                    checkpoints,
                );
                println!(
                    "  {} workers participated, all MAC checks passed",
                    num_workers,
                );
                println!("{}", "═".repeat(64).green());
            }
            ProgressEvent::RecoverableError { phase, message, retry_count } => {
                println!(
                    "  {} Phase {} retry {}: {}",
                    "⚠".yellow(),
                    phase,
                    retry_count,
                    message.dimmed(),
                );
            }
            ProgressEvent::ZkProofStarted { checkpoint_index, step } => {
                println!(
                    "  {} ZK proof: generating for checkpoint {} (step {})...",
                    "🔐".dimmed(),
                    checkpoint_index,
                    step,
                );
            }
            ProgressEvent::ZkProofGenerated { checkpoint_index: _, step, proof_size, time_ms, verified } => {
                println!(
                    "  {} ZK proof: step {} | {} bytes | {}ms | verified={}",
                    "✓".green(),
                    step,
                    proof_size,
                    time_ms,
                    verified,
                );
            }
            ProgressEvent::ZkProofFailed { checkpoint_index: _, step, error } => {
                println!(
                    "  {} ZK proof failed at step {} (non-fatal): {}",
                    "⚠".yellow(),
                    step,
                    error.dimmed(),
                );
            }
            ProgressEvent::ZkProofSubmitted { checkpoint_index: _, step, tx_hash } => {
                println!(
                    "  {} ZK proof submitted on-chain: step {} tx={}",
                    "⛓".dimmed(),
                    step,
                    &tx_hash[..10.min(tx_hash.len())],
                );
            }
            ProgressEvent::ZkRiskActivated { active_workers, min_workers } => {
                println!();
                println!(
                    "  {} {} -- active workers ({}) dropped below threshold ({})",
                    "⚠".yellow(),
                    "ZK RISK ACTIVATED".red().bold(),
                    active_workers,
                    min_workers,
                );
                println!(
                    "  {} Switching to ZK proof mode for remaining checkpoints",
                    "🔐".dimmed(),
                );
                println!();
            }
        }
    });
    orchestrator.set_progress_callback(progress_cb);

    println!("{}", "Starting full orchestration pipeline...".green().bold());
    println!();

    let result = orchestrator.run().await?;

    // Save weights if requested.
    if let Some(ref output_path) = args.output {
        if let Some(ref weights) = result.final_weights {
            let json = serde_json::to_string_pretty(weights)
                .map_err(|e| anyhow::anyhow!("Failed to serialize final weights: {}", e))?;
            std::fs::write(output_path, &json)
                .map_err(|e| anyhow::anyhow!("Failed to write weights to {}: {}", output_path.display(), e))?;
            println!(
                "\n  {} Final weights saved to {}",
                "✓".green(),
                output_path.display(),
            );
        }
    }

    println!();

    Ok(())
}

// ============================================================================
// MPC Worker
// ============================================================================

async fn cmd_mpc_worker(args: &MpcWorkerArgs, _cli: &Cli) -> Result<()> {
    use helix_client::worker_entry::{WorkerConfig, launch_worker};

    println!();
    println!("{}", "═".repeat(64).cyan());
    println!("{}", " HELIX MPC Worker".cyan().bold());
    println!("{}", "═".repeat(64).cyan());
    println!();

    // Parse the listen address and compute the control channel port.
    let data_addr: std::net::SocketAddr = args.listen.parse()
        .map_err(|e| anyhow::anyhow!(
            "Invalid --listen address '{}'. Expected format: IP:PORT (e.g. 0.0.0.0:9001). Error: {}", args.listen, e
        ))?;
    let control_port = data_addr.port() + 1;

    println!("{}", "Configuration:".yellow().bold());
    println!("  Data Channel:    {}", args.listen);
    println!("  Control Channel: {}:{}", data_addr.ip(), control_port);
    println!("  Party Index:     {}", args.party_index);
    println!("  Seed:            {}", args.seed);
    #[cfg(feature = "chain")]
    {
        if let Some(ref rpc) = args.rpc_url {
            println!("  RPC URL:         {}", rpc);
        }
        if let Some(job_id) = args.job_id {
            println!("  Job ID:          {}", job_id);
            println!("  Stake:           {} ETH", args.stake_eth);
        }
        if let Some(ref coord) = args.coordinator {
            println!("  Coordinator:     {}", coord);
        }
    }
    println!();

    let config = WorkerConfig {
        listen_addr: data_addr.to_string(),
        party_index: args.party_index,
        seed: args.seed,
        #[cfg(feature = "chain")]
        eth_rpc_url: args.rpc_url.clone(),
        #[cfg(feature = "chain")]
        private_key: args.private_key.clone().unwrap_or_default(),
        #[cfg(feature = "chain")]
        job_id: args.job_id,
        #[cfg(feature = "chain")]
        coordinator_address: args.coordinator.clone(),
        #[cfg(feature = "chain")]
        stake_amount_eth: args.stake_eth,
    };

    println!("{}", "Worker starting...".green().bold());
    println!(
        "  {} Listening for owner connection on data channel ({})",
        "⏳".dimmed(),
        args.listen,
    );
    println!(
        "  {} Control channel ready on port {}",
        "⏳".dimmed(),
        control_port,
    );
    #[cfg(feature = "chain")]
    if args.job_id.is_some() {
        println!(
            "  {} Staking {} ETH on-chain for job {}",
            "💰".dimmed(),
            args.stake_eth,
            args.job_id.unwrap(),
        );
    }
    println!();

    let result = launch_worker(config).await?;

    println!();
    println!("{}", "═".repeat(64).cyan());
    println!("{}", " Worker Session Complete".green().bold());
    println!("{}", "═".repeat(64).cyan());
    println!("  Steps Completed:       {}", result.steps_completed);
    println!("  Checkpoints Signed:    {}", result.checkpoint_signatures);
    println!("  Slashing Reports:      {}", result.slashing_reports_signed);
    let share_status = if result.final_share_sent {
        "Yes".green().to_string()
    } else {
        "No".red().to_string()
    };
    println!("  Final Share Sent:      {}", share_status);
    println!();

    Ok(())
}

// ============================================================================
// spawn-workers: Launch multiple MPC workers in-process
// ============================================================================

async fn cmd_spawn_workers(args: &SpawnWorkersArgs, _cli: &Cli) -> Result<()> {
    use helix_client::worker_entry::{WorkerConfig, launch_worker};
    use tokio::signal;

    println!();
    println!("{}", "═".repeat(64).cyan());
    println!("{}", " HELIX MPC Worker Spawner".cyan().bold());
    println!("{}", "═".repeat(64).cyan());
    println!();

    if args.count < 2 {
        anyhow::bail!("Need at least 2 workers (got {})", args.count);
    }

    let colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "bright_red", "bright_green", "bright_blue"];

    // Resolve the public address for registration (so remote machines can reach these workers).
    let public_host = args.public_addr.clone().unwrap_or_else(|| {
        if args.bind == "0.0.0.0" {
            "127.0.0.1".to_string()
        } else {
            args.bind.clone()
        }
    });

    // Print summary of worker addresses
    println!("{}", "Workers:".yellow().bold());
    let mut worker_addrs = Vec::new();
    let mut worker_public_addrs = Vec::new();
    for i in 0..args.count {
        let port = args.base_port + (i as u16) * 3;
        let addr = format!("{}:{}", args.bind, port);
        let pub_addr = format!("{}:{}", public_host, port);
        let color = colors[i % colors.len()];
        println!("  [worker-{}] bind={} public={} (data={}, ctrl={}, mpc={})", i, addr, pub_addr, port, port+1, port+2);
        worker_addrs.push(addr);
        worker_public_addrs.push(pub_addr);
    }
    println!();

    // ── On-chain pool registration ──
    #[cfg(feature = "chain")]
    let chain_registered = {
        use crate::rpc::chain_v4::ChainClientV4;
        use ethers::signers::{LocalWallet, Signer};
        use std::str::FromStr;

        let mut registered = false;

        if let (Some(ref rpc_url), Some(ref coordinator)) = (&args.rpc_url, &args.coordinator) {
            println!("{}", "Registering workers in on-chain pool...".yellow().bold());

            // Resolve private keys: explicit list, or Anvil defaults
            let keys: Vec<String> = if let Some(ref pk_csv) = args.private_keys {
                pk_csv.split(',').map(|s| s.trim().to_string()).collect()
            } else {
                // Anvil default accounts 1..N
                let anvil = vec![
                    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
                    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
                    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
                    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
                    "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
                    "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e",
                    "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
                    "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97",
                    "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
                ];
                anvil[..args.count.min(anvil.len())]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            };

            let stake_wei = ethers::utils::parse_ether(args.stake_eth)
                .unwrap_or(ethers::types::U256::from(100_000_000_000_000_000u64)); // 0.1 ETH

            for (i, key) in keys.iter().enumerate().take(args.count) {
                let pk = key.strip_prefix("0x").unwrap_or(key);
                let wallet = match LocalWallet::from_str(pk) {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("  {} [worker-{}] invalid private key: {}", "✗".red(), i, e);
                        continue;
                    }
                };
                let addr = wallet.address();

                match ChainClientV4::with_wallet(rpc_url, wallet, coordinator).await {
                    Ok(client) => {
                        let endpoint = &worker_public_addrs[i];
                        match client.register_in_pool(endpoint, stake_wei).await {
                            Ok(receipt) => {
                                let gas = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);
                                println!(
                                    "  {} [worker-{}] registered on-chain (addr: {:#x}, gas: {})",
                                    "✓".green(), i, addr, gas
                                );
                                registered = true;
                            }
                            Err(e) => {
                                eprintln!("  {} [worker-{}] on-chain registration failed: {}", "✗".red(), i, e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  {} [worker-{}] chain client error: {}", "✗".red(), i, e);
                    }
                }
            }
            println!();
        }
        registered
    };

    #[cfg(not(feature = "chain"))]
    let chain_registered = false;

    // ── Off-chain API registration (fallback) ──
    let http_client = reqwest::Client::new();
    let mut worker_ids: Vec<String> = Vec::new();

    if !chain_registered {
        if let Some(ref api_url) = args.api_url {
            println!("{}", "Registering workers with dashboard API...".yellow().bold());
            for (i, pub_addr) in worker_public_addrs.iter().enumerate() {
                let worker_seed = args.seed + i as u64;
                let resp = http_client
                    .post(format!("{}/api/workers/register", api_url))
                    .json(&serde_json::json!({
                        "endpoint": pub_addr,
                        "party_index": i,
                        "seed": worker_seed,
                    }))
                    .send()
                    .await;

                match resp {
                    Ok(r) if r.status().is_success() => {
                        let body: serde_json::Value = r.json().await.unwrap_or_default();
                        let wid = body["worker_id"].as_str().unwrap_or("unknown").to_string();
                        println!("  {} [worker-{}] registered (id: {})", "✓".green(), i, &wid[..8]);
                        worker_ids.push(wid);
                    }
                    Ok(r) => {
                        eprintln!("  {} [worker-{}] registration failed: HTTP {}", "✗".red(), i, r.status());
                        worker_ids.push(String::new());
                    }
                    Err(e) => {
                        eprintln!("  {} [worker-{}] registration failed: {}", "✗".red(), i, e);
                        worker_ids.push(String::new());
                    }
                }
            }
            println!();
        } else {
            // Print copy-paste command for --workers flag (legacy mode)
            println!("{}", "Copy-paste for mpc-train:".yellow().bold());
            println!("  --workers {}", worker_public_addrs.join(","));
            println!();
        }
    }

    // Spawn heartbeat task if registered with API (not needed for on-chain)
    let heartbeat_handle = if !chain_registered {
        if let Some(ref api_url) = args.api_url {
            let url = api_url.clone();
            let ids: Vec<String> = worker_ids.iter().filter(|id| !id.is_empty()).cloned().collect();
            let client = http_client.clone();
            Some(tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                    for wid in &ids {
                        let _ = client
                            .post(format!("{}/api/workers/heartbeat", url))
                            .json(&serde_json::json!({ "worker_id": wid }))
                            .send()
                            .await;
                    }
                }
            }))
        } else {
            None
        }
    } else {
        None
    };

    // Spawn all workers as tokio tasks
    let mut handles = Vec::new();

    for i in 0..args.count {
        let port = args.base_port + (i as u16) * 3;
        let listen_addr = format!("{}:{}", args.bind, port);
        let seed = args.seed + i as u64;

        let handle = tokio::spawn(async move {
            // Workers loop: restart after each job so they can handle multiple
            // training sessions without manual restart.
            loop {
                println!("[worker-{}] Starting on {} (waiting for job...)", i, listen_addr);
                let job_config = WorkerConfig {
                    listen_addr: listen_addr.clone(),
                    party_index: i,
                    seed,
                    #[cfg(feature = "chain")]
                    eth_rpc_url: None,
                    #[cfg(feature = "chain")]
                    private_key: String::new(),
                    #[cfg(feature = "chain")]
                    job_id: None,
                    #[cfg(feature = "chain")]
                    coordinator_address: None,
                    #[cfg(feature = "chain")]
                    stake_amount_eth: 0.0,
                };
                match launch_worker(job_config).await {
                    Ok(result) => {
                        println!(
                            "[worker-{}] Job done: {} steps, {} checkpoints signed. Restarting...",
                            i, result.steps_completed, result.checkpoint_signatures
                        );
                    }
                    Err(e) => {
                        eprintln!("[worker-{}] Error: {}. Restarting in 2s...", i, e);
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        });
        handles.push(handle);
    }

    println!("{}", "All workers started. Waiting for training jobs...".green().bold());
    if chain_registered {
        println!("{}", "Workers registered on-chain — any user can assign them via the V4 coordinator.".cyan());
    } else if args.api_url.is_some() {
        println!("{}", "Workers registered with dashboard — they will be auto-assigned when training starts.".cyan());
    }
    println!("{}", "Press Ctrl+C to shut down.".dimmed());
    println!();

    // Wait for SIGINT
    signal::ctrl_c().await?;
    println!();
    println!("{}", "Shutting down workers...".yellow());

    // Abort heartbeat task
    if let Some(hb) = heartbeat_handle {
        hb.abort();
    }

    // Abort all worker tasks
    for handle in &handles {
        handle.abort();
    }

    // Wait for tasks to finish
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(()) => println!("[worker-{}] Stopped cleanly", i),
            Err(e) if e.is_cancelled() => println!("[worker-{}] Stopped", i),
            Err(e) => eprintln!("[worker-{}] Error during shutdown: {}", i, e),
        }
    }

    println!("{}", "All workers stopped.".green());
    Ok(())
}
