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
mod runner;

use clap::Parser;
use tracing_subscriber::EnvFilter;

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

    /// Enable ZK proof generation at checkpoints (not yet implemented for Stage 10)
    #[arg(long)]
    pub zk_proofs: bool,

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

    if args.zk_proofs {
        display::warn("--zk-proofs flag noted but ZK proof generation is not yet wired for Stage 10.");
        display::info("MPC training will proceed with MAC verification only.");
        println!();
    }

    let demo = runner::DemoRunner::new(args);
    demo.run().await
}
