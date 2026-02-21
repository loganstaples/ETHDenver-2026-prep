//! HELIX MPC Training Demo Binary
//!
//! Demonstrates the full HELIX MPC-primary training pipeline on MNIST:
//!
//! 1. Generate synthetic MNIST dataset (784-dim inputs, 10 classes)
//! 2. Create N MPC workers with in-memory transport mesh
//! 3. Distribute model weights as additive secret shares
//! 4. Initialize SPDZ MAC authentication for integrity
//! 5. Pre-generate Beaver triples for secure multiplication
//! 6. Train with MAC-verified gradient steps
//! 7. Optionally simulate and detect a cheating party
//! 8. Evaluate accuracy on held-out test set
//! 9. Display full training summary with metrics

mod display;
mod evaluator;
mod risk;
mod runner;
mod zk_prover;

use clap::Parser;
use tracing_subscriber::EnvFilter;

/// ZK proof generation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkMode {
    /// No ZK proofs.
    Off,
    /// ZK proof at every checkpoint.
    Always,
    /// ZK proofs activate automatically when risk is detected.
    Risk,
}

fn parse_zk_mode(s: &str) -> Result<ZkMode, String> {
    match s.to_lowercase().as_str() {
        "off" => Ok(ZkMode::Off),
        "always" => Ok(ZkMode::Always),
        "risk" => Ok(ZkMode::Risk),
        _ => Err(format!("invalid ZK mode '{}': expected off, always, or risk", s)),
    }
}

/// HELIX MPC Training Demo - ETHDenver 2026
///
/// Trains a 784->32->10 MLP on synthetic MNIST data using multi-party
/// computation with SPDZ MAC verification. Model weights remain secret-shared
/// across all workers throughout training. Every computation is verified
/// through information-theoretic MACs.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "helix-demo",
    about = "HELIX: Trustless Distributed ML Training with MPC",
    version
)]
pub struct Args {
    /// Number of training steps
    #[arg(short = 'n', long, default_value = "200")]
    pub steps: usize,

    /// Number of MPC worker parties
    #[arg(short = 'w', long, default_value = "3")]
    pub workers: usize,

    /// Checkpoint frequency (MAC verify + commitment every N steps)
    #[arg(short = 'c', long, default_value = "100")]
    pub checkpoint_freq: u64,

    /// Simulate a cheating party at the midpoint of training
    #[arg(long)]
    pub simulate_cheater: bool,

    /// ZK proof mode: off (no ZK), always (every checkpoint), risk (auto-activate on threat).
    #[arg(long, default_value = "off", value_parser = parse_zk_mode)]
    pub zk_mode: ZkMode,

    /// Minimum worker count before risk-based ZK activates (only for --zk-mode risk).
    #[arg(long, default_value = "2")]
    pub min_workers_for_mpc: usize,

    /// Verbose logging output
    #[arg(short, long)]
    pub verbose: bool,

    /// Learning rate for SGD
    #[arg(long, default_value = "0.01")]
    pub lr: f64,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of training samples to generate
    #[arg(long, default_value = "1000")]
    pub train_size: usize,

    /// Number of test samples to generate
    #[arg(long, default_value = "200")]
    pub test_size: usize,

    /// Use real MNIST data (downloads and caches on first run)
    #[arg(long)]
    pub real_mnist: bool,

    /// Custom directory for MNIST data cache
    #[arg(long)]
    pub mnist_cache_dir: Option<String>,

    /// Skip on-chain checkpoint settlement phase.
    /// By default, the demo deploys HelixCoordinatorV4 to a local Anvil node,
    /// registers workers, and submits multi-party signed checkpoint attestations.
    /// Pass this flag to run MPC training only without on-chain submission.
    #[arg(long)]
    pub skip_chain: bool,
}

impl Args {
    /// Input dimension (MNIST = 784).
    pub fn d_in(&self) -> usize { 784 }
    /// Hidden dimension.
    pub fn d_hid(&self) -> usize { 32 }
    /// Output dimension (MNIST = 10 classes).
    pub fn d_out(&self) -> usize { 10 }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.verbose {
        "helix_demo=debug,helix_mpc=debug"
    } else {
        "helix_demo=info,helix_mpc=warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_target(false)
        .init();

    match args.zk_mode {
        ZkMode::Off => {}
        ZkMode::Always => {
            display::info("ZK proofs ENABLED: StateTransitionCircuit proof at every checkpoint");
            println!();
        }
        ZkMode::Risk => {
            display::info(&format!(
                "ZK proofs in RISK mode: auto-activate when workers < {} or cheater detected",
                args.min_workers_for_mpc,
            ));
            println!();
        }
    }

    let demo = runner::DemoRunner::new(args);
    demo.run().await
}
