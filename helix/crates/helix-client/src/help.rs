//! HELIX CLI Help and Documentation
//!
//! Provides comprehensive help text, examples, and documentation for all CLI commands.
//! Includes:
//! - Detailed command descriptions
//! - Usage examples with explanations
//! - Configuration reference
//! - Troubleshooting guides
//! - Quick start tutorials

use colored::*;
use std::collections::HashMap;

// ============================================================================
// Help System
// ============================================================================

/// Main help system for the HELIX CLI
pub struct HelpSystem {
    /// Command help entries
    commands: HashMap<String, CommandHelp>,
    /// Topic help entries
    topics: HashMap<String, TopicHelp>,
}

impl HelpSystem {
    /// Create a new help system with all documentation
    pub fn new() -> Self {
        let mut system = Self {
            commands: HashMap::new(),
            topics: HashMap::new(),
        };

        system.register_commands();
        system.register_topics();
        system
    }

    /// Print general help
    pub fn print_general_help(&self) {
        println!("{}", BANNER.cyan());
        println!();
        println!("{}", "HELIX - Trustless Distributed ML Training Protocol".bold());
        println!();
        println!("{}", "USAGE:".yellow().bold());
        println!("    helix [OPTIONS] <COMMAND>");
        println!();
        println!("{}", "OPTIONS:".yellow().bold());
        println!("    -c, --config <FILE>     Configuration file path [default: helix.toml]");
        println!("    -v, --verbose           Increase verbosity (-v, -vv, -vvv)");
        println!("        --format <FORMAT>   Output format (text, json) [default: text]");
        println!("    -h, --help              Print help information");
        println!("    -V, --version           Print version information");
        println!();
        println!("{}", "COMMANDS:".yellow().bold());
        println!("    {}       Initialize a new HELIX node or network", "init".green());
        println!("    {}      Start training with model config", "train".green());
        println!("    {}       Join an existing training network", "join".green());
        println!("    {}     Query node/network status", "status".green());
        println!("    {}      Query model or round data", "query".green());
        println!("    {}     Export training artifacts", "export".green());
        println!("    {}       Start a multi-node demo network", "demo".green());
        println!("    {}  Orchestrate distributed training across nodes", "orchestrate".green());
        println!("    {}     Run health checks", "health".green());
        println!("    {}  Start the status dashboard server", "dashboard".green());
        println!("    {}  Run performance benchmarks", "benchmark".green());
        println!("    {}       Aggregate and view logs", "logs".green());
        println!("    {}      Watch and reload configuration", "watch".green());
        println!("    {}  Launch interactive training visualization", "visualize".green());
        println!();
        println!("{}", "QUICK START:".yellow().bold());
        println!("    1. Initialize a node:     helix init --network local");
        println!("    2. Start training:        helix train --config model.toml");
        println!("    3. Or run demo:           helix demo --scenario quick");
        println!("    4. View status:           helix status --watch");
        println!();
        println!("{}", "HELP TOPICS:".yellow().bold());
        println!("    helix help getting-started    Quick start guide");
        println!("    helix help configuration      Configuration reference");
        println!("    helix help profiles           Deployment profiles");
        println!("    helix help training           Training workflow");
        println!("    helix help proofs             Proof system overview");
        println!("    helix help troubleshooting    Common issues and solutions");
        println!();
        println!("Run '{}' for more information on a specific command.", "helix <command> --help".cyan());
        println!("Run '{}' for detailed documentation on a topic.", "helix help <topic>".cyan());
    }

    /// Print command-specific help
    pub fn print_command_help(&self, command: &str) {
        if let Some(help) = self.commands.get(command) {
            help.print();
        } else {
            eprintln!("{} Unknown command: {}", "Error:".red().bold(), command);
            eprintln!("Run 'helix --help' for a list of available commands.");
        }
    }

    /// Print topic-specific help
    pub fn print_topic_help(&self, topic: &str) {
        if let Some(help) = self.topics.get(topic) {
            help.print();
        } else {
            eprintln!("{} Unknown topic: {}", "Error:".red().bold(), topic);
            eprintln!("Available topics: getting-started, configuration, profiles, training, proofs, troubleshooting");
        }
    }

    fn register_commands(&mut self) {
        // Init command
        self.commands.insert("init".to_string(), CommandHelp {
            name: "init".to_string(),
            short_desc: "Initialize a new HELIX node or network".to_string(),
            long_desc: r#"
Initialize a new HELIX node with the specified configuration. This command creates
the necessary configuration files, generates a wallet if requested, and sets up
the data directories for node operation.

The init command supports multiple network profiles:
  - local:   Local development with Anvil/Hardhat
  - anvil:   Anvil testnet fork
  - sepolia: Sepolia testnet
  - mainnet: Ethereum mainnet (production)
"#.to_string(),
            usage: "helix init [OPTIONS]".to_string(),
            options: vec![
                ("--name <NAME>".to_string(), "Node name (auto-generated if not specified)".to_string()),
                ("-N, --network <NETWORK>".to_string(), "Network to initialize (local, anvil, sepolia, mainnet) [default: local]".to_string()),
                ("--generate-wallet".to_string(), "Generate a new wallet keypair".to_string()),
                ("--aggregator".to_string(), "Initialize as an aggregator node".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix init".to_string(),
                    description: "Initialize with default settings (local network)".to_string(),
                },
                Example {
                    command: "helix init --name my-node --network sepolia".to_string(),
                    description: "Initialize for Sepolia testnet with custom name".to_string(),
                },
                Example {
                    command: "helix init --generate-wallet --aggregator".to_string(),
                    description: "Initialize as aggregator with new wallet".to_string(),
                },
            ],
            see_also: vec!["train".to_string(), "join".to_string(), "config".to_string()],
        });

        // Train command
        self.commands.insert("train".to_string(), CommandHelp {
            name: "train".to_string(),
            short_desc: "Start training with model configuration".to_string(),
            long_desc: r#"
Start a distributed training session using a TOML configuration file. The train
command loads model architecture, hyperparameters, data sources, and proof settings
from the config file and begins the training process.

Training modes:
  - Normal:   Full training with proof generation and on-chain verification
  - Dry-run:  Simulate training without compute or on-chain interaction

The command provides real-time progress reporting including:
  - Current round and total rounds
  - Loss values and convergence metrics
  - Proof generation status
  - Worker participation statistics
  - Error bound tracking

Use --dry-run to validate configuration and test workflow before committing
resources to actual training.
"#.to_string(),
            usage: "helix train [OPTIONS]".to_string(),
            options: vec![
                ("-C, --config <FILE>".to_string(), "Training configuration file [default: model.toml]".to_string()),
                ("--model-id <ID>".to_string(), "Override model ID from config".to_string()),
                ("--max-rounds <N>".to_string(), "Override maximum training rounds".to_string()),
                ("--learning-rate <RATE>".to_string(), "Override learning rate".to_string()),
                ("--dry-run".to_string(), "Simulate training without real compute".to_string()),
                ("--resume <CHECKPOINT>".to_string(), "Resume from checkpoint".to_string()),
                ("--workers <N>".to_string(), "Minimum workers required to start".to_string()),
                ("--timeout <SECS>".to_string(), "Round timeout in seconds".to_string()),
                ("--output-dir <PATH>".to_string(), "Output directory for artifacts".to_string()),
                ("--no-progress".to_string(), "Disable progress bar output".to_string()),
                ("--watch".to_string(), "Continue watching after training completes".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix train --config model.toml".to_string(),
                    description: "Start training with configuration file".to_string(),
                },
                Example {
                    command: "helix train --config model.toml --dry-run".to_string(),
                    description: "Validate config with simulated training".to_string(),
                },
                Example {
                    command: "helix train -C model.toml --max-rounds 50 --learning-rate 0.001".to_string(),
                    description: "Start training with overridden hyperparameters".to_string(),
                },
                Example {
                    command: "helix train --resume checkpoint_round_10.json".to_string(),
                    description: "Resume training from checkpoint".to_string(),
                },
            ],
            see_also: vec!["init".to_string(), "join".to_string(), "status".to_string(), "export".to_string()],
        });

        // Join command
        self.commands.insert("join".to_string(), CommandHelp {
            name: "join".to_string(),
            short_desc: "Join an existing training network".to_string(),
            long_desc: r#"
Join a HELIX training network to participate in distributed model training.
This command connects to the network coordinator, stakes tokens, and begins
participating in training rounds.

Before joining, ensure you have:
  1. Initialized your node (helix init)
  2. Sufficient ETH for staking
  3. Network connectivity to the coordinator
"#.to_string(),
            usage: "helix join [OPTIONS] --coordinator <ADDR> --model-id <ID>".to_string(),
            options: vec![
                ("-c, --coordinator <ADDR>".to_string(), "Address of the network coordinator".to_string()),
                ("-m, --model-id <ID>".to_string(), "Model ID to join training for".to_string()),
                ("-s, --stake <AMOUNT>".to_string(), "Stake amount in ETH".to_string()),
                ("--capabilities <CAPS>".to_string(), "Node capabilities (train, aggregate, prove)".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix join -c 0x5FbDB... -m 0 -s 1.0".to_string(),
                    description: "Join model 0 training with 1 ETH stake".to_string(),
                },
                Example {
                    command: "helix join -c 0x5FbDB... -m 0 --capabilities train,prove".to_string(),
                    description: "Join as worker with specific capabilities".to_string(),
                },
            ],
            see_also: vec!["init".to_string(), "status".to_string(), "query".to_string()],
        });

        // Status command
        self.commands.insert("status".to_string(), CommandHelp {
            name: "status".to_string(),
            short_desc: "Query node and network status".to_string(),
            long_desc: r#"
Display the current status of your node and the connected network. Shows:
  - Node health and uptime
  - Connected peers
  - Current training progress
  - Staking status
  - Proof submission status

Use --watch for continuous monitoring with automatic refresh.
"#.to_string(),
            usage: "helix status [OPTIONS]".to_string(),
            options: vec![
                ("-d, --detailed".to_string(), "Show detailed status information".to_string()),
                ("-w, --watch".to_string(), "Watch status continuously".to_string()),
                ("-m, --model-id <ID>".to_string(), "Filter by specific model ID".to_string()),
                ("--interval <SECS>".to_string(), "Refresh interval for watch mode [default: 5]".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix status".to_string(),
                    description: "Show current status".to_string(),
                },
                Example {
                    command: "helix status --detailed --watch".to_string(),
                    description: "Watch detailed status continuously".to_string(),
                },
                Example {
                    command: "helix status -m 0 --format json".to_string(),
                    description: "Get model 0 status as JSON".to_string(),
                },
            ],
            see_also: vec!["query".to_string(), "health".to_string(), "dashboard".to_string()],
        });

        // Query command
        self.commands.insert("query".to_string(), CommandHelp {
            name: "query".to_string(),
            short_desc: "Query model, round, stake, or proof data".to_string(),
            long_desc: r#"
Query detailed information about models, training rounds, stakes, and proofs
from the HELIX network. Supports multiple query types:
  - model:      Model configuration and current state
  - round:      Specific round details and commitments
  - stake:      Staking information for addresses
  - proof:      Proof details and verification status
  - error:      Error bound accumulation
  - worker:     Worker node statistics
  - aggregator: Aggregator node statistics
  - metrics:    Overall network metrics
"#.to_string(),
            usage: "helix query <TYPE> [OPTIONS] --model-id <ID>".to_string(),
            options: vec![
                ("<TYPE>".to_string(), "Query type: model, round, stake, proof, error".to_string()),
                ("-m, --model-id <ID>".to_string(), "Model ID to query".to_string()),
                ("-r, --round-id <ID>".to_string(), "Round ID (for round queries)".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix query model -m 0".to_string(),
                    description: "Query model 0 information".to_string(),
                },
                Example {
                    command: "helix query round -m 0 -r 42".to_string(),
                    description: "Query round 42 of model 0".to_string(),
                },
                Example {
                    command: "helix query stake -m 0".to_string(),
                    description: "Query your stake for model 0".to_string(),
                },
                Example {
                    command: "helix query error -m 0 --format json".to_string(),
                    description: "Query error bounds as JSON".to_string(),
                },
            ],
            see_also: vec!["status".to_string(), "export".to_string()],
        });

        // Export command
        self.commands.insert("export".to_string(), CommandHelp {
            name: "export".to_string(),
            short_desc: "Export trained models, proofs, and analytics".to_string(),
            long_desc: r#"
Export training artifacts from the HELIX network. Supports exporting:
  - model:      Trained model weights and architecture
  - proofs:     ZK proofs for all rounds
  - metrics:    Training metrics and statistics
  - logs:       Node and training logs
  - checkpoint: Training checkpoints for resumption
  - analytics:  Comprehensive analytics report
  - all:        All artifacts

Exports can be formatted as JSON, CSV, or binary (for models).
"#.to_string(),
            usage: "helix export <TYPE> [OPTIONS] --model-id <ID>".to_string(),
            options: vec![
                ("<TYPE>".to_string(), "Export type: model, proofs, metrics, logs, all".to_string()),
                ("-m, --model-id <ID>".to_string(), "Model ID to export".to_string()),
                ("-o, --output <PATH>".to_string(), "Output directory [default: ./export]".to_string()),
                ("--format <FMT>".to_string(), "Export format (json, csv, binary) [default: json]".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix export model -m 0 -o ./models".to_string(),
                    description: "Export model 0 weights".to_string(),
                },
                Example {
                    command: "helix export all -m 0 --format json".to_string(),
                    description: "Export all artifacts as JSON".to_string(),
                },
                Example {
                    command: "helix export analytics -m 0".to_string(),
                    description: "Generate analytics report".to_string(),
                },
            ],
            see_also: vec!["query".to_string(), "status".to_string()],
        });

        // Demo command
        self.commands.insert("demo".to_string(), CommandHelp {
            name: "demo".to_string(),
            short_desc: "Start a multi-node demo network".to_string(),
            long_desc: r#"
Run a demonstration of the HELIX protocol with simulated nodes. Available scenarios:
  - quick:           Fast demo showing basic training (3 rounds)
  - full-training:   Complete training cycle with proofs (10 rounds)
  - slashing:        Demonstrates fault detection and slashing
  - multi-model:     Concurrent training of multiple models
  - fault-tolerance: Network resilience under node failures

The demo deploys contracts, starts worker nodes, performs staking, and runs
training rounds with proof generation and verification.
"#.to_string(),
            usage: "helix demo [OPTIONS] [SCENARIO]".to_string(),
            options: vec![
                ("[SCENARIO]".to_string(), "Demo scenario [default: full-training]".to_string()),
                ("-w, --workers <N>".to_string(), "Number of worker nodes [default: 3]".to_string()),
                ("-r, --rounds <N>".to_string(), "Number of training rounds [default: 5]".to_string()),
                ("--round-duration <SECS>".to_string(), "Round duration in seconds [default: 60]".to_string()),
                ("--headless".to_string(), "Run without interactive UI".to_string()),
                ("--skip-deploy".to_string(), "Skip contract deployment".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix demo".to_string(),
                    description: "Run default full-training demo".to_string(),
                },
                Example {
                    command: "helix demo quick".to_string(),
                    description: "Run quick demo (fastest)".to_string(),
                },
                Example {
                    command: "helix demo slashing -w 5".to_string(),
                    description: "Run slashing demo with 5 workers".to_string(),
                },
                Example {
                    command: "helix demo fault-tolerance --headless".to_string(),
                    description: "Run fault tolerance demo without UI".to_string(),
                },
            ],
            see_also: vec!["visualize".to_string(), "orchestrate".to_string()],
        });

        // Orchestrate command
        self.commands.insert("orchestrate".to_string(), CommandHelp {
            name: "orchestrate".to_string(),
            short_desc: "Orchestrate distributed training across nodes".to_string(),
            long_desc: r#"
Manage a network of HELIX nodes for distributed training. Actions:
  - start:    Start the network with specified nodes
  - stop:     Stop all network nodes gracefully
  - restart:  Restart the network
  - scale:    Scale the network to a new node count
  - topology: Display network topology
"#.to_string(),
            usage: "helix orchestrate <ACTION> [OPTIONS]".to_string(),
            options: vec![
                ("<ACTION>".to_string(), "Orchestration action: start, stop, restart, scale, topology".to_string()),
                ("-n, --nodes <N>".to_string(), "Number of nodes [default: 3]".to_string()),
                ("-t, --template <FILE>".to_string(), "Configuration template".to_string()),
                ("--network-name <NAME>".to_string(), "Network name [default: helix-demo]".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix orchestrate start -n 5".to_string(),
                    description: "Start network with 5 nodes".to_string(),
                },
                Example {
                    command: "helix orchestrate scale -n 10".to_string(),
                    description: "Scale network to 10 nodes".to_string(),
                },
                Example {
                    command: "helix orchestrate topology".to_string(),
                    description: "Display network topology".to_string(),
                },
            ],
            see_also: vec!["demo".to_string(), "status".to_string()],
        });

        // Health command
        self.commands.insert("health".to_string(), CommandHelp {
            name: "health".to_string(),
            short_desc: "Run health checks on node and network".to_string(),
            long_desc: r#"
Perform health checks on the HELIX node and connected services. Targets:
  - all:       Run all health checks
  - nodes:     Check node processes and resources
  - contracts: Verify smart contract connectivity
  - network:   Check RPC and peer connectivity
  - storage:   Verify storage and IPFS connectivity
"#.to_string(),
            usage: "helix health [OPTIONS] [TARGET]".to_string(),
            options: vec![
                ("[TARGET]".to_string(), "Health check target [default: all]".to_string()),
                ("-t, --timeout <SECS>".to_string(), "Check timeout [default: 10]".to_string()),
                ("-d, --detailed".to_string(), "Show detailed health information".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix health".to_string(),
                    description: "Run all health checks".to_string(),
                },
                Example {
                    command: "helix health contracts -d".to_string(),
                    description: "Check contracts with details".to_string(),
                },
            ],
            see_also: vec!["status".to_string(), "logs".to_string()],
        });

        // Visualize command
        self.commands.insert("visualize".to_string(), CommandHelp {
            name: "visualize".to_string(),
            short_desc: "Launch interactive training visualization".to_string(),
            long_desc: r#"
Launch a terminal-based UI for monitoring HELIX training in real-time.
The visualization includes:
  - Training progress and loss curves
  - Worker status dashboard
  - Proof generation timeline
  - Network topology view
  - Event log streaming

Navigation:
  - Tab/1-4: Switch between views
  - ↑↓: Scroll event log
  - ←→: Select worker
  - P: Pause/resume updates
  - Q: Quit visualization
"#.to_string(),
            usage: "helix visualize [OPTIONS] [SCENARIO]".to_string(),
            options: vec![
                ("[SCENARIO]".to_string(), "Demo scenario to visualize [default: quick]".to_string()),
                ("-w, --workers <N>".to_string(), "Number of worker nodes [default: 5]".to_string()),
                ("-r, --rounds <N>".to_string(), "Number of training rounds [default: 10]".to_string()),
                ("--refresh-rate <MS>".to_string(), "UI refresh rate in milliseconds [default: 100]".to_string()),
                ("--attach".to_string(), "Attach to existing demo without starting new one".to_string()),
            ],
            examples: vec![
                Example {
                    command: "helix visualize".to_string(),
                    description: "Start visualization with quick demo".to_string(),
                },
                Example {
                    command: "helix visualize full-training -w 5 -r 10".to_string(),
                    description: "Visualize full training with 5 workers".to_string(),
                },
                Example {
                    command: "helix visualize --attach".to_string(),
                    description: "Attach to existing running demo".to_string(),
                },
            ],
            see_also: vec!["demo".to_string(), "status".to_string(), "dashboard".to_string()],
        });
    }

    fn register_topics(&mut self) {
        // Getting Started topic
        self.topics.insert("getting-started".to_string(), TopicHelp {
            title: "Getting Started with HELIX".to_string(),
            content: GETTING_STARTED.to_string(),
        });

        // Configuration topic
        self.topics.insert("configuration".to_string(), TopicHelp {
            title: "HELIX Configuration Reference".to_string(),
            content: CONFIGURATION.to_string(),
        });

        // Profiles topic
        self.topics.insert("profiles".to_string(), TopicHelp {
            title: "Deployment Profiles".to_string(),
            content: PROFILES.to_string(),
        });

        // Training topic
        self.topics.insert("training".to_string(), TopicHelp {
            title: "Training Workflow".to_string(),
            content: TRAINING.to_string(),
        });

        // Proofs topic
        self.topics.insert("proofs".to_string(), TopicHelp {
            title: "Proof System Overview".to_string(),
            content: PROOFS.to_string(),
        });

        // Troubleshooting topic
        self.topics.insert("troubleshooting".to_string(), TopicHelp {
            title: "Troubleshooting Guide".to_string(),
            content: TROUBLESHOOTING.to_string(),
        });
    }
}

impl Default for HelpSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Help Structures
// ============================================================================

/// Help information for a command
struct CommandHelp {
    name: String,
    short_desc: String,
    long_desc: String,
    usage: String,
    options: Vec<(String, String)>,
    examples: Vec<Example>,
    see_also: Vec<String>,
}

impl CommandHelp {
    fn print(&self) {
        println!("{}", format!("helix-{}", self.name).cyan().bold());
        println!("{}", self.short_desc);
        println!();
        println!("{}", self.long_desc.trim());
        println!();
        println!("{}", "USAGE:".yellow().bold());
        println!("    {}", self.usage);
        println!();

        if !self.options.is_empty() {
            println!("{}", "OPTIONS:".yellow().bold());
            for (opt, desc) in &self.options {
                println!("    {:30} {}", opt.green(), desc);
            }
            println!();
        }

        if !self.examples.is_empty() {
            println!("{}", "EXAMPLES:".yellow().bold());
            for example in &self.examples {
                println!("    {}", example.command.cyan());
                println!("        {}", example.description.dimmed());
                println!();
            }
        }

        if !self.see_also.is_empty() {
            println!("{}", "SEE ALSO:".yellow().bold());
            let refs: Vec<_> = self.see_also.iter().map(|s| format!("helix-{}", s)).collect();
            println!("    {}", refs.join(", "));
        }
    }
}

/// Example command with description
struct Example {
    command: String,
    description: String,
}

/// Help information for a topic
struct TopicHelp {
    title: String,
    content: String,
}

impl TopicHelp {
    fn print(&self) {
        println!("{}", self.title.cyan().bold());
        println!("{}", "=".repeat(self.title.len()).cyan());
        println!();
        println!("{}", self.content.trim());
    }
}

// ============================================================================
// Help Content
// ============================================================================

const BANNER: &str = r#"
  _   _ _____ _     _____  __
 | | | | ____| |   |_ _\ \/ /
 | |_| |  _| | |    | | \  /
 |  _  | |___| |___ | | /  \
 |_| |_|_____|_____|___/_/\_\
"#;

const GETTING_STARTED: &str = r#"
This guide will help you get started with HELIX, the trustless distributed
ML training protocol.

PREREQUISITES
-------------
- Rust toolchain (1.70+)
- Node.js (for frontend/tooling)
- An Ethereum wallet with testnet ETH

INSTALLATION
------------
1. Clone the repository:
   git clone https://github.com/helix-protocol/helix.git
   cd helix

2. Build the CLI:
   cargo build --release -p helix-client

3. Add to PATH:
   export PATH="$PATH:./target/release"

QUICK START
-----------
1. Initialize a node:
   helix init --name my-node --network local

2. Run a quick demo to see HELIX in action:
   helix demo quick

3. For interactive visualization:
   helix visualize

NEXT STEPS
----------
- Read 'helix help configuration' for config options
- Read 'helix help training' for training workflow
- Read 'helix help proofs' for proof system details
"#;

const CONFIGURATION: &str = r#"
HELIX is configured via TOML files. The default location is ./helix.toml.

CONFIGURATION FILE STRUCTURE
----------------------------

[node]
name = "helix-node-abcd1234"     # Unique node identifier
network = "local"                 # Network: local, sepolia, mainnet
data_dir = "~/.helix/data"       # Data storage directory
capabilities = ["train", "prove"] # Node capabilities
role = "Worker"                  # Worker, Aggregator, or Coordinator

[network]
listen_addr = "0.0.0.0:9000"     # P2P listen address
bootstrap_nodes = []              # Bootstrap peer addresses
enable_mdns = true                # Local peer discovery
max_peers = 50                    # Maximum peer connections

[training]
batch_size = 32                   # Training batch size
learning_rate = 0.001             # Learning rate
max_rounds = 100                  # Maximum training rounds
round_timeout = 120               # Round timeout (seconds)

[rpc]
url = "http://localhost:8545"     # Ethereum RPC endpoint
chain_id = 31337                  # Chain ID
timeout = 30                      # Request timeout (seconds)

[contracts]
coordinator = "0x5FbDB..."        # HelixCoordinator address
verifier = "0xe7f17..."           # HelixVerifier address

[staking]
min_stake = 0.1                   # Minimum stake (ETH)
default_stake = 0.5               # Default stake amount
lock_period = 604800              # Lock period (seconds)

[proof]
proof_system = "Approximate"      # Approximate, Full, or Optimistic
error_bound_max = 1000.0          # Maximum error bound
verification_timeout = 60         # Verification timeout

ENVIRONMENT VARIABLES
---------------------
HELIX_CONFIG      - Config file path (default: helix.toml)
HELIX_DATA_DIR    - Data directory override
HELIX_RPC_URL     - RPC endpoint override
HELIX_PRIVATE_KEY - Wallet private key (for automated deployments)
"#;

const PROFILES: &str = r#"
HELIX provides pre-configured profiles for different deployment scenarios.

AVAILABLE PROFILES
------------------

LOCAL (--network local)
  - For local development with Anvil/Hardhat
  - Fast block times, unlimited ETH
  - mDNS peer discovery enabled
  - Debug logging enabled
  - No stake lock period

ANVIL (--network anvil)
  - For Anvil testnet fork
  - Mainnet state for realistic testing
  - 1-hour stake lock period
  - Info-level logging

SEPOLIA (--network sepolia)
  - For Sepolia testnet deployment
  - Real network conditions
  - 1-day stake lock period
  - Gradient compression enabled
  - Requires testnet ETH

MAINNET (--network mainnet)
  - For production deployment
  - 7-day stake lock period
  - Conservative timeouts
  - Minimal logging
  - USE WITH CAUTION

USAGE
-----
Initialize with a profile:
  helix init --network sepolia

Generate profile configs:
  helix config generate-profiles --output ./profiles

Switch profiles at runtime:
  helix --config ./profiles/sepolia.toml status
"#;

const TRAINING: &str = r#"
HELIX TRAINING WORKFLOW
-----------------------

OVERVIEW
--------
HELIX enables trustless distributed ML training where:
1. Multiple workers train on local data shards
2. Gradients are aggregated using MPC for privacy
3. Each computation is verified with approximate ZK proofs
4. Economic incentives ensure honest participation

QUICK START
-----------
Start training with a configuration file:
  helix train --config model.toml

Test your config without real compute:
  helix train --config model.toml --dry-run

Resume from checkpoint:
  helix train --resume checkpoint_round_10.json

CONFIGURATION
-------------
Training is configured via TOML files. Generate an example:
  helix init --network local
  # Creates ~/.helix/model.toml with example config

Key configuration sections:
  [model]      - Architecture, layers, precision
  [training]   - Learning rate, batch size, rounds
  [data]       - Data paths and preprocessing
  [proof]      - Proof system and error bounds
  [network]    - Coordinator and worker settings

WORKFLOW
--------
1. MODEL REGISTRATION
   The model owner registers a model on-chain with:
   - Initial model weights (IPFS hash)
   - Training configuration
   - Minimum stake requirement
   - Maximum error bound

2. WORKER JOINING
   Workers join by:
   - Staking tokens as collateral
   - Specifying capabilities (train, aggregate, prove)
   - Connecting to the coordinator

3. TRAINING ROUNDS
   Each round consists of:
   a. Gradient computation (workers train on local data)
   b. Gradient aggregation (MPC for privacy)
   c. Proof generation (approximate ZK proof)
   d. Proof verification (on-chain)
   e. Model update (new commitment on-chain)

4. FINALIZATION
   After training completes:
   - Final model is uploaded to IPFS
   - Stakes are released (minus any slashing)
   - Model is marked complete on-chain

ERROR BOUNDS
------------
HELIX uses approximate proofs with bounded error:
- Each round adds some error to computation
- Total error is accumulated and tracked on-chain
- If error exceeds max bound, round is rejected
- This allows efficient proofs while maintaining integrity

SLASHING
--------
Workers can be slashed for:
- Submitting invalid proofs
- Exceeding error bounds
- Failing to submit in time
- Byzantine behavior detected by other workers
"#;

const PROOFS: &str = r#"
HELIX PROOF SYSTEM
------------------

OVERVIEW
--------
HELIX uses approximate ZK proofs to verify ML computations efficiently.
Unlike exact ZK proofs, approximate proofs allow small bounded errors
while still providing strong security guarantees.

PROOF TYPES
-----------

APPROXIMATE (Default)
  - Allows bounded computational error
  - Much faster than exact proofs
  - Error tracked and accumulated on-chain
  - Suitable for most training scenarios

FULL
  - Exact ZK proofs with no error
  - Significantly slower generation
  - Higher gas costs for verification
  - For high-stakes applications

OPTIMISTIC
  - No upfront proof required
  - Fraud proofs if dispute raised
  - Challenge period before finalization
  - Most gas-efficient for honest parties

PROOF LIFECYCLE
---------------
1. GENERATION
   - Worker computes training step
   - Generates proof of correct computation
   - Includes error bound commitment

2. SUBMISSION
   - Proof submitted to aggregator
   - Aggregator batches multiple proofs
   - Submitted on-chain for verification

3. VERIFICATION
   - Smart contract verifies proof
   - Checks error bound not exceeded
   - Updates model state if valid

4. COMMITMENT
   - New model commitment stored on-chain
   - Error bound accumulated
   - Round marked complete

CONFIGURATION
-------------
[proof]
proof_system = "Approximate"   # Approximate, Full, Optimistic
error_bound_max = 1000.0       # Maximum total error allowed
verification_timeout = 60      # Timeout for verification (seconds)
batch_proofs = true            # Batch multiple proofs together
proof_compression = false      # Compress proofs for storage
"#;

const TROUBLESHOOTING: &str = r#"
TROUBLESHOOTING GUIDE
---------------------

COMMON ISSUES
-------------

"Connection refused" when joining network
  - Ensure the coordinator address is correct
  - Check firewall allows outbound connections
  - Verify RPC endpoint is accessible
  Solution: helix health network

"Insufficient stake" error
  - Your wallet doesn't have enough ETH
  - Check required stake: helix query stake -m <MODEL_ID>
  Solution: Fund your wallet with sufficient ETH

"Proof verification failed"
  - Computation exceeded error bounds
  - Network congestion caused timeout
  Solution: Check error bounds, retry with longer timeout

"Peer discovery timeout"
  - No bootstrap nodes configured
  - Network isolation (firewall/NAT)
  Solution: Add bootstrap nodes or enable mDNS for local

"Contract not found" error
  - Wrong network or chain ID
  - Contracts not deployed
  Solution: Verify network config matches deployed contracts

Node crashes during training
  - Out of memory (model too large)
  - Disk space exhausted
  Solution: Increase resources or reduce batch size

DIAGNOSTICS
-----------

Check system health:
  helix health --detailed

View node logs:
  helix logs -f --level debug

Check network connectivity:
  helix health network

Verify contract deployment:
  helix health contracts

GETTING HELP
------------

For additional support:
  - GitHub Issues: https://github.com/helix-protocol/helix/issues
  - Discord: https://discord.gg/helix
  - Documentation: https://docs.helix.network
"#;

// ============================================================================
// Public API Functions
// ============================================================================

/// Print the CLI banner
pub fn print_banner() {
    println!("{}", BANNER.cyan());
}

/// Print a quick reference card
pub fn print_quick_reference() {
    println!("{}", "HELIX Quick Reference".cyan().bold());
    println!("{}", "=".repeat(50).cyan());
    println!();
    println!("{}", "Essential Commands:".yellow().bold());
    println!("  helix init                Initialize node");
    println!("  helix train -C model.toml Start training");
    println!("  helix demo quick          Run quick demo");
    println!("  helix status --watch      Monitor status");
    println!("  helix visualize           Interactive UI");
    println!();
    println!("{}", "Demo Scenarios:".yellow().bold());
    println!("  quick          Fast 3-round demo");
    println!("  full-training  Complete 10-round training");
    println!("  slashing       Fault detection demo");
    println!("  multi-model    Concurrent training");
    println!("  fault-tolerance Network resilience");
    println!();
    println!("{}", "Profiles:".yellow().bold());
    println!("  local    Development (fast, free)");
    println!("  anvil    Testnet fork");
    println!("  sepolia  Public testnet");
    println!("  mainnet  Production (careful!)");
    println!();
    println!("{}", "For more help:".dimmed());
    println!("  helix --help              General help");
    println!("  helix <cmd> --help        Command help");
    println!("  helix help <topic>        Topic help");
}

/// Print version information
pub fn print_version() {
    println!("{} {}", "helix".cyan().bold(), env!("CARGO_PKG_VERSION"));
    println!("HELIX CLI - Trustless Distributed ML Training");
    println!();
    println!("Authors:  {}", env!("CARGO_PKG_AUTHORS"));
    println!("License:  MIT/Apache-2.0");
    println!("Homepage: https://helix.network");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_help_system_creation() {
        let system = HelpSystem::new();
        assert!(!system.commands.is_empty());
        assert!(!system.topics.is_empty());
    }

    #[test]
    fn test_command_exists() {
        let system = HelpSystem::new();
        assert!(system.commands.contains_key("init"));
        assert!(system.commands.contains_key("train"));
        assert!(system.commands.contains_key("join"));
        assert!(system.commands.contains_key("demo"));
        assert!(system.commands.contains_key("visualize"));
    }

    #[test]
    fn test_topic_exists() {
        let system = HelpSystem::new();
        assert!(system.topics.contains_key("getting-started"));
        assert!(system.topics.contains_key("configuration"));
        assert!(system.topics.contains_key("troubleshooting"));
    }
}
