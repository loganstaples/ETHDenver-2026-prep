//! HELIX End-to-End Demo Binary
//!
//! Runs the full HELIX trustless ML training pipeline as a single command:
//! 1. Spawn Anvil (local Ethereum node)
//! 2. Deploy contracts (MockVerifier + HelixCoordinatorV2)
//! 3. Register model and start training
//! 4. Run 3 worker threads with real ZK proof generation
//! 5. Submit proofs on-chain
//! 6. Show loss curve and training progress
//! 7. Demonstrate adversarial detection and slashing

mod chain;
mod display;
mod srs;
mod bench;
mod worker;

use clap::Parser;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

/// HELIX End-to-End Demo
#[derive(Parser, Debug)]
#[command(name = "helix-demo", about = "Run the full HELIX E2E demo")]
struct Args {
    /// Number of training steps per worker
    #[arg(short = 'n', long, default_value = "20")]
    steps: usize,

    /// Number of worker threads
    #[arg(short = 'w', long, default_value = "3")]
    workers: usize,

    /// Learning rate
    #[arg(long, default_value = "0.01")]
    lr: f64,

    /// Model hidden dimension
    #[arg(long, default_value = "2")]
    d_hid: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Skip on-chain deployment (train only, no Anvil)
    #[arg(long)]
    offline: bool,

    /// Run performance benchmark after demo
    #[arg(long)]
    bench: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip slashing demo
    #[arg(long)]
    no_slash: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.verbose {
        "helix_demo=debug,helix_prover=info,helix_node=info"
    } else {
        "helix_demo=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_target(false)
        .init();

    let demo_start = Instant::now();

    display::banner();

    // ── Phase 1: Shared SRS ──────────────────────────────────────────────
    display::phase(1, "SHARED TRUSTED SETUP", "Generating consistent SRS for all workers");
    let srs_start = Instant::now();
    let srs_path = srs::ensure_shared_srs()?;
    display::success(&format!(
        "SRS cached at {} ({:.1}s)",
        srs_path.display(),
        srs_start.elapsed().as_secs_f64()
    ));

    // ── Phase 2: On-Chain Setup ──────────────────────────────────────────
    let chain_env = if !args.offline {
        display::phase(2, "ON-CHAIN INFRASTRUCTURE", "Deploying contracts to local Anvil");
        let env = chain::setup_chain().await?;
        display::success(&format!(
            "Anvil running on {}",
            env.anvil_endpoint
        ));
        display::success(&format!(
            "MockVerifier:    {}",
            display::short_addr(&env.mock_verifier_addr)
        ));
        display::success(&format!(
            "Coordinator:     {}",
            display::short_addr(&env.coordinator_addr)
        ));

        // Register model
        let model_id = chain::register_model(&env).await?;
        display::success(&format!("Model registered (ID: {})", model_id));

        // Stake for workers
        chain::stake_for_workers(&env, model_id, args.workers).await?;
        display::success(&format!(
            "{} workers staked (1 ETH each)",
            args.workers
        ));

        // Start training round
        chain::start_round(&env, model_id).await?;
        display::success("Training round started on-chain");

        Some(env)
    } else {
        display::phase(2, "OFFLINE MODE", "Skipping chain deployment");
        display::info("Training will run without on-chain verification");
        None
    };

    // ── Phase 3: Distributed Training ────────────────────────────────────
    display::phase(
        3,
        "DISTRIBUTED TRAINING",
        &format!(
            "{} workers x {} steps (lr={}, d_hid={})",
            args.workers, args.steps, args.lr, args.d_hid
        ),
    );

    let dataset = worker::make_dataset();
    display::info(&format!(
        "Dataset: {} samples (y = x1 + x2 regression)",
        dataset.len()
    ));

    // Run workers
    let worker_results = worker::run_workers(
        args.workers,
        args.steps,
        args.d_hid,
        args.lr,
        args.seed,
        &dataset,
    )
    .await?;

    // Aggregate results
    let mut all_losses: Vec<Vec<f64>> = Vec::new();
    let mut total_proofs = 0usize;
    let mut total_prove_time_ms = 0u128;

    for (i, result) in worker_results.iter().enumerate() {
        all_losses.push(result.losses.clone());
        total_proofs += result.proofs_generated;
        total_prove_time_ms += result.total_prove_time.as_millis();

        display::success(&format!(
            "Worker {}: {} steps, loss {:.4} -> {:.4}, {} proofs ({:.0}ms avg)",
            i,
            result.steps_completed,
            result.losses.first().unwrap_or(&0.0),
            result.losses.last().unwrap_or(&0.0),
            result.proofs_generated,
            if result.proofs_generated > 0 {
                result.total_prove_time.as_millis() as f64 / result.proofs_generated as f64
            } else {
                0.0
            },
        ));
    }

    // ── Phase 4: On-Chain Proof Submission ────────────────────────────────
    if let Some(ref env) = chain_env {
        display::phase(4, "ON-CHAIN VERIFICATION", "Submitting proofs to smart contract");

        let submit_count = std::cmp::min(5, worker_results.iter().map(|r| r.evm_bundles.len()).sum());
        let mut submitted = 0;

        for (worker_idx, result) in worker_results.iter().enumerate() {
            for bundle in &result.evm_bundles {
                if submitted >= submit_count {
                    break;
                }
                match chain::submit_proof(env, bundle).await {
                    Ok(gas) => {
                        display::success(&format!(
                            "Proof #{} (worker {}) verified on-chain (gas: {})",
                            submitted, worker_idx, gas
                        ));
                        submitted += 1;
                    }
                    Err(e) => {
                        display::warn(&format!(
                            "Proof submission failed: {} (continuing)",
                            e
                        ));
                    }
                }
            }
        }
        display::metric(&format!("{}/{} proofs submitted on-chain", submitted, submit_count));
    }

    // ── Phase 5: Slashing Demo ───────────────────────────────────────────
    if !args.no_slash {
        if let Some(ref env) = chain_env {
            display::phase(5, "ADVERSARIAL DETECTION", "Demonstrating Byzantine fault handling");
            match chain::demonstrate_slashing(env).await {
                Ok(()) => {
                    display::success("Malicious proof detected and rejected");
                    display::success("Worker stake slashed via smart contract");
                }
                Err(e) => {
                    display::warn(&format!("Slashing demo: {}", e));
                }
            }
        }
    }

    // ── Phase 6: Loss Curve & Summary ────────────────────────────────────
    display::phase(6, "TRAINING RESULTS", "Loss convergence and metrics");
    display::loss_curve(&all_losses);

    let avg_loss_start: f64 = all_losses
        .iter()
        .filter_map(|l| l.first())
        .sum::<f64>()
        / all_losses.len() as f64;
    let avg_loss_end: f64 = all_losses
        .iter()
        .filter_map(|l| l.last())
        .sum::<f64>()
        / all_losses.len() as f64;

    println!();
    let reduction_pct = if avg_loss_start > 1e-10 {
        (1.0 - avg_loss_end / avg_loss_start) * 100.0
    } else {
        0.0
    };
    display::metric(&format!(
        "Avg loss:       {:.6} -> {:.6} ({:.1}% reduction)",
        avg_loss_start, avg_loss_end, reduction_pct,
    ));
    display::metric(&format!(
        "Total proofs:   {} across {} workers",
        total_proofs, args.workers
    ));
    display::metric(&format!(
        "Avg proof time: {:.0}ms",
        if total_proofs > 0 {
            total_prove_time_ms as f64 / total_proofs as f64
        } else {
            0.0
        }
    ));
    display::metric(&format!(
        "Total time:     {:.1}s",
        demo_start.elapsed().as_secs_f64()
    ));

    // ── Phase 7: Benchmark (optional) ────────────────────────────────────
    if args.bench {
        display::phase(7, "PERFORMANCE BENCHMARK", "Measuring ZK overhead vs native training");
        let bench_result = bench::run_overhead_benchmark(args.d_hid, &dataset)?;
        bench::display_results(&bench_result);
    }

    // ── Summary ──────────────────────────────────────────────────────────
    display::summary(
        demo_start.elapsed(),
        total_proofs,
        args.workers,
        avg_loss_start,
        avg_loss_end,
        chain_env.is_some(),
    );

    // Cleanup: drop chain_env to kill Anvil
    drop(chain_env);

    Ok(())
}
