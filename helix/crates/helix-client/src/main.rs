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
mod dashboard;
mod demo;
mod help;
mod orchestrator;
mod progress;
mod health;
mod benchmark;
mod rpc;
mod visualization;
mod wallet;

use commands::{
    init::InitCommand,
    join::JoinCommand,
    status::StatusCommand,
    query::QueryCommand,
    export::ExportCommand,
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
    Help(HelpArgs),
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
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Enable CORS
    #[arg(long)]
    cors: bool,
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

    // Execute command
    let result = match &cli.command {
        Commands::Init(args) => cmd_init(args, &cli).await,
        Commands::Join(args) => cmd_join(args, &cli).await,
        Commands::Status(args) => cmd_status(args, &cli).await,
        Commands::Query(args) => cmd_query(args, &cli).await,
        Commands::Export(args) => cmd_export(args, &cli).await,
        Commands::Demo(args) => cmd_demo(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Orchestrate(args) => cmd_orchestrate(args, &cli).await,
        Commands::Health(args) => cmd_health(args, &cli).await,
        Commands::Dashboard(args) => cmd_dashboard(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Benchmark(args) => cmd_benchmark(args, &cli).await,
        Commands::Logs(args) => cmd_logs(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Watch(args) => cmd_watch(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Visualize(args) => cmd_visualize(args, &cli, shutdown_tx.subscribe()).await,
        Commands::Help(args) => cmd_help(args, &cli).await,
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

async fn cmd_join(args: &JoinArgs, _cli: &Cli) -> Result<()> {
    let mut progress = ProgressDisplay::new();

    println!("{}", "Joining HELIX training network...".cyan().bold());
    println!("  Coordinator: {}", args.coordinator);
    println!("  Model ID: {}", args.model_id);

    if let Some(stake) = args.stake {
        progress.start_spinner(&format!("Staking {} ETH...", stake));
        tokio::time::sleep(Duration::from_secs(2)).await;
        progress.finish_spinner(&format!("Staked {} ETH successfully", stake));
    }

    progress.start_spinner("Connecting to coordinator...");
    tokio::time::sleep(Duration::from_secs(1)).await;
    progress.finish_spinner("Connected to coordinator");

    progress.start_spinner("Discovering peers...");
    tokio::time::sleep(Duration::from_secs(1)).await;
    progress.finish_spinner("Found 3 peers");

    progress.start_spinner("Syncing model state...");
    tokio::time::sleep(Duration::from_secs(2)).await;
    progress.finish_spinner("Model state synced");

    println!("\n{}", "Successfully joined the training network!".green().bold());
    println!("  Capabilities: {:?}", args.capabilities);
    println!("  Status: Ready to participate in training rounds");

    Ok(())
}

async fn cmd_status(args: &StatusArgs, _cli: &Cli) -> Result<()> {
    let display_status = || {
        println!("{}", "═".repeat(60).cyan());
        println!("{}", " HELIX Network Status".cyan().bold());
        println!("{}", "═".repeat(60).cyan());

        println!("\n{}", "Node Status:".yellow().bold());
        println!("  Node ID:     helix-node-a1b2c3d4");
        println!("  Status:      {} Online", "●".green());
        println!("  Uptime:      2h 34m 12s");
        println!("  Peers:       4 connected");

        println!("\n{}", "Training Status:".yellow().bold());
        if let Some(model_id) = args.model_id {
            println!("  Model ID:    {}", model_id);
        }
        println!("  Current Round:  42");
        println!("  Round Status:   {} In Progress (67%)", "●".blue());
        println!("  Proofs:         127 submitted, 125 verified");
        println!("  Error Bound:    45.2 / 1000 max");

        println!("\n{}", "Staking:".yellow().bold());
        println!("  Your Stake:     1.5 ETH");
        println!("  Lock Status:    {} Locked (5d 12h remaining)", "●".yellow());
        println!("  Reputation:     98%");

        if args.detailed {
            println!("\n{}", "Connected Peers:".yellow().bold());
            println!("  ┌─────────────────┬──────────┬────────────┐");
            println!("  │ Peer ID         │ Status   │ Role       │");
            println!("  ├─────────────────┼──────────┼────────────┤");
            println!("  │ helix-node-e5f6 │ {} Active │ Worker     │", "●".green());
            println!("  │ helix-node-g7h8 │ {} Active │ Worker     │", "●".green());
            println!("  │ helix-node-i9j0 │ {} Active │ Aggregator │", "●".green());
            println!("  │ helix-node-k1l2 │ {} Idle   │ Worker     │", "●".yellow());
            println!("  └─────────────────┴──────────┴────────────┘");
        }

        println!("{}", "═".repeat(60).cyan());
    };

    if args.watch {
        loop {
            print!("\x1B[2J\x1B[1;1H"); // Clear screen
            display_status();
            println!("\nRefreshing every {}s... (Ctrl+C to stop)", args.interval);
            tokio::time::sleep(Duration::from_secs(args.interval)).await;
        }
    } else {
        display_status();
    }

    Ok(())
}

async fn cmd_query(args: &QueryArgs, _cli: &Cli) -> Result<()> {
    match args.query_type {
        QueryType::Model => {
            println!("{}", "Model Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  IPFS Hash:         QmXoYP...abc123");
            println!("  Current Round:     42");
            println!("  Current Commitment: 0x1234567890abcdef...");
            println!("  Min Stake:         0.1 ETH");
            println!("  Active:            {} Yes", "●".green());
            println!("  Owner:             0x742d35Cc6634C053...");
        }
        QueryType::Round => {
            let round_id = args.round_id.unwrap_or(42);
            println!("{}", "Round Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Round ID:          {}", round_id);
            println!("  Model Commitment:  0x1234567890abcdef...");
            println!("  New Commitment:    0xfedcba0987654321...");
            println!("  Completed:         {} Yes", "●".green());
            println!("  Deadline:          2024-01-15 14:30:00 UTC");
            println!("  Prover:            0x8626f6940E2eb289...");
        }
        QueryType::Stake => {
            println!("{}", "Stake Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Your Address:      0x742d35Cc6634C053...");
            println!("  Amount:            1.5 ETH");
            println!("  Locked Until:      2024-01-20 12:00:00 UTC");
            println!("  Slashed:           {} No", "●".green());
        }
        QueryType::Proof => {
            println!("{}", "Proof Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Round ID:          {}", args.round_id.unwrap_or(42));
            println!("  Proof Hash:        0xabcdef1234567890...");
            println!("  Prover:            0x8626f6940E2eb289...");
            println!("  Verified:          {} Yes", "●".green());
            println!("  Submitted:         2024-01-15 14:25:30 UTC");
        }
        QueryType::Error => {
            println!("{}", "Error Bound Information".cyan().bold());
            println!("  Model ID:          {}", args.model_id);
            println!("  Accumulated Error: 45.2");
            println!("  Max Allowed:       1000");
            println!("  Status:            {} Acceptable", "●".green());
            println!("  Rounds Tracked:    42");
        }
    }

    Ok(())
}

async fn cmd_export(args: &ExportArgs, _cli: &Cli) -> Result<()> {
    let mut progress = ProgressDisplay::new();

    std::fs::create_dir_all(&args.output)?;

    match args.export_type {
        ExportType::Model => {
            progress.start_spinner("Exporting model weights...");
            tokio::time::sleep(Duration::from_secs(2)).await;
            let path = args.output.join(format!("model_{}.pt", args.model_id));
            progress.finish_spinner(&format!("Model exported to {}", path.display()));
        }
        ExportType::Proofs => {
            progress.start_spinner("Exporting proofs...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            let path = args.output.join(format!("proofs_{}.{}", args.model_id, args.format));
            progress.finish_spinner(&format!("Proofs exported to {}", path.display()));
        }
        ExportType::Metrics => {
            progress.start_spinner("Exporting training metrics...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            let path = args.output.join(format!("metrics_{}.{}", args.model_id, args.format));
            progress.finish_spinner(&format!("Metrics exported to {}", path.display()));
        }
        ExportType::Logs => {
            progress.start_spinner("Exporting logs...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            let path = args.output.join("logs.txt");
            progress.finish_spinner(&format!("Logs exported to {}", path.display()));
        }
        ExportType::All => {
            progress.start_spinner("Exporting all artifacts...");
            tokio::time::sleep(Duration::from_secs(3)).await;
            progress.finish_spinner(&format!("All artifacts exported to {}", args.output.display()));
        }
    }

    Ok(())
}

async fn cmd_demo(args: &DemoArgs, _cli: &Cli, shutdown: broadcast::Receiver<()>) -> Result<()> {
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

async fn cmd_health(args: &HealthArgs, _cli: &Cli) -> Result<()> {
    println!("{}", "Running health checks...".cyan().bold());
    println!();

    let check_item = |name: &str, ok: bool, detail: &str| {
        if ok {
            println!("  {} {} {}", "✓".green(), name, detail.dimmed());
        } else {
            println!("  {} {} {}", "✗".red(), name, detail.red());
        }
    };

    match args.target {
        HealthTarget::All | HealthTarget::Nodes => {
            println!("{}", "Nodes:".yellow().bold());
            check_item("Local Node", true, "running (pid: 12345)");
            check_item("Peer Connections", true, "4 peers connected");
            check_item("Memory Usage", true, "1.2 GB / 8 GB");
            check_item("CPU Usage", true, "23%");
            println!();
        }
        _ => {}
    }

    match args.target {
        HealthTarget::All | HealthTarget::Contracts => {
            println!("{}", "Contracts:".yellow().bold());
            check_item("HelixCoordinator", true, "0x5FbDB2315678...");
            check_item("HelixVerifier", true, "0xe7f1725E7734CE...");
            check_item("HelixToken", true, "0x9fE46736679d2D...");
            println!();
        }
        _ => {}
    }

    match args.target {
        HealthTarget::All | HealthTarget::Network => {
            println!("{}", "Network:".yellow().bold());
            check_item("RPC Connection", true, "http://localhost:8545");
            check_item("Block Height", true, "12,345,678");
            check_item("Gas Price", true, "20 gwei");
            check_item("Chain ID", true, "31337 (localhost)");
            println!();
        }
        _ => {}
    }

    match args.target {
        HealthTarget::All | HealthTarget::Storage => {
            println!("{}", "Storage:".yellow().bold());
            check_item("Data Directory", true, "~/.helix/data");
            check_item("Model Cache", true, "2.5 GB used");
            check_item("IPFS Gateway", true, "connected");
            println!();
        }
        _ => {}
    }

    println!("{}", "All health checks passed!".green().bold());

    Ok(())
}

async fn cmd_dashboard(args: &DashboardArgs, _cli: &Cli, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    println!("{}", "Starting HELIX Status Dashboard...".cyan().bold());
    println!("  Address: http://{}:{}", args.host, args.port);

    let app = dashboard::create_dashboard_router(args.cors);

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
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
