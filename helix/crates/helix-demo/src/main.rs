//! HELIX End-to-End Demo Binary
//!
//! Runs the full HELIX trustless ML training pipeline as a single command:
//! 1. Spawn Anvil (local Ethereum node)
//! 2. Deploy real Halo2Verifier + HelixCoordinatorV2
//! 3. Register model and start training
//! 4. Run worker threads with real ZK proof generation
//! 5. Submit proofs on-chain (verified by real BN254 pairing checks)
//! 6. Run aggregator to collect and validate proofs
//! 7. Optionally run MPC mode (--mpc)
//! 8. Demonstrate adversarial detection and slashing
//! 9. Show loss curve and training progress
//! 10. Optionally run comprehensive benchmarks (--bench)

mod aggregator;
mod bench;
mod chain;
mod display;
mod mpc;
mod srs;
mod worker;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use helix_core::{DatasetSource, ModelArchitecture};
use helix_prover::provers::training_prover_v2::MLTrainingProverV2;
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

    /// Enable MPC mode (distribute weights across parties)
    #[arg(long)]
    mpc: bool,

    /// Number of MPC parties (used with --mpc)
    #[arg(long, default_value = "3")]
    mpc_parties: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip slashing demo
    #[arg(long)]
    no_slash: bool,

    /// Save benchmark results as JSON to this path
    #[arg(long)]
    bench_json: Option<PathBuf>,

    /// Path to benchmark baseline for regression detection
    #[arg(long)]
    bench_baseline: Option<PathBuf>,

    /// Dataset source: "builtin" or path to a CSV file
    #[arg(long, default_value = "builtin")]
    dataset: String,

    /// Feature column indices (comma-separated, for CSV datasets)
    #[arg(long, default_value = "0,1")]
    feature_cols: String,

    /// Label column indices (comma-separated, for CSV datasets)
    #[arg(long, default_value = "2")]
    label_cols: String,

    /// Path to model architecture config (JSON file)
    #[arg(long)]
    model_config: Option<PathBuf>,

    /// Checkpoint proving interval: generate ZK proof every N steps (1 = every step).
    /// Higher values reduce proving overhead by using MPC consensus between checkpoints.
    #[arg(long, default_value = "1")]
    checkpoint_interval: u64,
}

/// Parses CLI dataset args into a `DatasetSource`.
fn parse_dataset_source(args: &Args) -> DatasetSource {
    if args.dataset == "builtin" {
        DatasetSource::Builtin
    } else {
        let feature_columns: Vec<usize> = args
            .feature_cols
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let label_columns: Vec<usize> = args
            .label_cols
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        DatasetSource::LocalCsv {
            path: args.dataset.clone(),
            feature_columns,
            label_columns,
        }
    }
}

/// Loads model architecture from JSON file or returns default from CLI args.
fn load_model_config(args: &Args) -> anyhow::Result<ModelArchitecture> {
    if let Some(ref path) = args.model_config {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read model config '{}': {}", path.display(), e))?;
        let arch: ModelArchitecture = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse model config: {}", e))?;
        Ok(arch)
    } else {
        Ok(ModelArchitecture {
            d_in: 2,
            d_hid: args.d_hid,
            d_out: 1,
            activation: "relu".to_string(),
            num_layers: 1,
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let filter = if args.verbose {
        "helix_demo=debug,helix_prover=info,helix_node=info,helix_mpc=info"
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

    // Parse dataset source and model architecture from CLI args
    let dataset_source = parse_dataset_source(&args);
    let arch = load_model_config(&args)?;

    // ── Phase 1b: Extract VK for on-chain deployment ─────────────────────
    let vk_data = if !args.offline {
        display::info("Initializing prover to extract VK for on-chain verifier...");
        let prover = MLTrainingProverV2::new(arch.d_in, arch.d_hid, arch.d_out);
        let vk = prover.export_vk_data()
            .map_err(|e| anyhow::anyhow!("Failed to export VK data: {}", e))?;
        display::success(&format!(
            "VK extracted: s_g2[0]={:.20}...",
            &vk.s_g2.0[..vk.s_g2.0.len().min(20)],
        ));
        Some(vk)
    } else {
        None
    };

    // ── Phase 2: On-Chain Setup ──────────────────────────────────────────
    let chain_env = if let Some(ref vk) = vk_data {
        display::phase(2, "ON-CHAIN INFRASTRUCTURE", "Deploying real Halo2Verifier to local Anvil");
        let env = chain::setup_chain(vk).await?;
        display::success(&format!(
            "Anvil running on {}",
            env.anvil_endpoint
        ));
        display::success(&format!(
            "Halo2Verifier:   {}",
            display::short_addr(&env.verifier_addr)
        ));
        display::success(&format!(
            "Coordinator:     {}",
            display::short_addr(&env.coordinator_addr)
        ));

        // Register model with real serialized weights and IPFS CID
        let init_model = worker::create_small_model_pub(arch.d_in, arch.d_hid, arch.d_out, args.seed);
        let ckpt_bytes = init_model.to_checkpoint(0).to_bytes()
            .map_err(|e| anyhow::anyhow!("Failed to serialize model checkpoint: {}", e))?;
        let initial_commitment = init_model.commitment();
        let (model_id, cid) = chain::register_model_with_weights(&env, &ckpt_bytes, initial_commitment).await?;
        display::success(&format!("Model registered (ID: {}, CID: {})", model_id, cid));

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
    let checkpoint_desc = if args.checkpoint_interval > 1 {
        format!(", checkpoint every {} steps", args.checkpoint_interval)
    } else {
        String::new()
    };
    display::phase(
        3,
        "DISTRIBUTED TRAINING",
        &format!(
            "{} workers x {} steps (lr={}, arch={}x{}x{}{})",
            args.workers, args.steps, args.lr, arch.d_in, arch.d_hid, arch.d_out, checkpoint_desc
        ),
    );

    let dataset = worker::load_dataset(&dataset_source)?;
    let dataset_desc = match &dataset_source {
        DatasetSource::Builtin => "builtin (y = x1 + x2 regression)".to_string(),
        DatasetSource::LocalCsv { path, .. } => format!("CSV: {}", path),
        DatasetSource::Ipfs { cid } => format!("IPFS: {}", cid),
        DatasetSource::Http { url } => format!("HTTP: {}", url),
    };
    display::info(&format!(
        "Dataset: {} samples ({})",
        dataset.len(), dataset_desc
    ));

    // Run workers
    let worker_results = worker::run_workers(
        args.workers,
        args.steps,
        arch.d_hid,
        args.lr,
        args.seed,
        &dataset,
        args.checkpoint_interval,
    )
    .await?;

    // Aggregate results
    let mut all_losses: Vec<Vec<f64>> = Vec::new();
    let mut total_proofs = 0usize;
    let mut total_verified = 0usize;
    let mut total_cache_hits = 0usize;
    let mut total_prove_time_ms = 0u128;

    for (i, result) in worker_results.iter().enumerate() {
        all_losses.push(result.losses.clone());
        total_proofs += result.proofs_generated;
        total_verified += result.proofs_verified;
        total_cache_hits += result.cache_hits;
        total_prove_time_ms += result.total_prove_time.as_millis();

        display::success(&format!(
            "Worker {}: {} steps, loss {:.4} -> {:.4}, {} proofs ({} verified, {:.0}ms avg)",
            i,
            result.steps_completed,
            result.losses.first().unwrap_or(&0.0),
            result.losses.last().unwrap_or(&0.0),
            result.proofs_generated,
            result.proofs_verified,
            if result.proofs_generated > 0 {
                result.total_prove_time.as_millis() as f64 / result.proofs_generated as f64
            } else {
                0.0
            },
        ));
    }

    // ── Phase 4: On-Chain Proof Submission ────────────────────────────────
    let mut gas_costs = Vec::new();
    if let Some(ref env) = chain_env {
        display::phase(4, "ON-CHAIN VERIFICATION", "Submitting proofs to real Halo2Verifier");

        let submit_count = std::cmp::min(
            5,
            worker_results.iter().map(|r| r.evm_bundles.len()).sum(),
        );
        let mut submitted = 0;

        for (worker_idx, result) in worker_results.iter().enumerate() {
            for bundle in &result.evm_bundles {
                if submitted >= submit_count {
                    break;
                }
                match chain::submit_proof(env, bundle).await {
                    Ok(gas) => {
                        display::success(&format!(
                            "Proof #{} (worker {}) verified on-chain via BN254 pairing (gas: {})",
                            submitted, worker_idx, gas
                        ));
                        gas_costs.push(gas);
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
        display::metric(&format!(
            "{}/{} proofs submitted and verified on-chain",
            submitted, submit_count
        ));
    }

    // ── Phase 4b: Aggregator Verification ────────────────────────────────
    if !args.offline {
        display::phase(
            5,
            "AGGREGATOR VERIFICATION",
            "Collecting proofs, validating, and aggregating commitments",
        );

        match aggregator::run_aggregator_demo(&worker_results).await {
            Ok(agg_result) => {
                display::metric(&format!(
                    "Aggregator: {}/{} proofs accepted, {} invalid rejected",
                    agg_result.proofs_accepted,
                    agg_result.proofs_submitted,
                    agg_result.invalid_rejected,
                ));
            }
            Err(e) => {
                display::warn(&format!("Aggregator demo: {}", e));
            }
        }
    }

    // ── Phase 5b: MPC Mode ───────────────────────────────────────────────
    if args.mpc {
        display::phase(
            6,
            "MPC TRAINING",
            &format!(
                "{} parties, {} steps with additive secret sharing",
                args.mpc_parties, args.steps,
            ),
        );

        match mpc::run_mpc_training(
            args.mpc_parties,
            arch.d_hid,
            args.lr,
            args.seed,
            &dataset,
            args.steps.min(5), // Limit MPC steps for demo speed
            false,             // Skip MPC proofs in demo
        )
        .await
        {
            Ok(mpc_result) => {
                display::metric(&format!(
                    "MPC loss trajectory: {:.4} -> {:.4}",
                    mpc_result.losses.first().unwrap_or(&0.0),
                    mpc_result.losses.last().unwrap_or(&0.0),
                ));
            }
            Err(e) => {
                display::warn(&format!("MPC training: {}", e));
            }
        }
    }

    // ── Phase 6: Slashing Demo ───────────────────────────────────────────
    if !args.no_slash {
        if let Some(ref env) = chain_env {
            display::phase(
                7,
                "ADVERSARIAL DETECTION",
                "Demonstrating Byzantine fault handling with real verifier",
            );
            match chain::demonstrate_slashing(env).await {
                Ok(()) => {
                    display::success("Malicious proof detected and rejected by real Halo2Verifier");
                    display::success("Worker stake slashed via smart contract");
                }
                Err(e) => {
                    display::warn(&format!("Slashing demo: {}", e));
                }
            }
        }
    }

    // ── Phase 7: Loss Curve & Summary ────────────────────────────────────
    display::phase(8, "TRAINING RESULTS", "Loss convergence and metrics");
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
        "Total proofs:   {} ({} self-verified) across {} workers",
        total_proofs, total_verified, args.workers
    ));
    if total_cache_hits > 0 {
        display::metric(&format!("Cache hits:     {}", total_cache_hits));
    }
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

    // ── Phase 8: Benchmark (optional) ────────────────────────────────────
    if args.bench {
        display::phase(
            9,
            "PERFORMANCE BENCHMARK",
            "Comprehensive ZK overhead analysis with regression detection",
        );
        let mut bench_result = bench::run_overhead_benchmark(arch.d_hid, &dataset)?;

        // Add gas costs from on-chain submissions
        if !gas_costs.is_empty() {
            bench::add_gas_costs(&mut bench_result, &gas_costs);
        }

        bench::display_results(&bench_result);

        // Save JSON output if requested
        if let Some(ref json_path) = args.bench_json {
            bench::save_json(&bench_result, json_path)?;
        }

        // Regression detection
        if let Some(ref baseline_path) = args.bench_baseline {
            display::info("Checking for performance regressions...");
            let passed = bench::check_regression(&bench_result, baseline_path);
            if !passed {
                display::warn("Performance regression detected! Check baseline.");
            }
            // Always save current as new baseline
            bench::save_baseline(&bench_result, baseline_path)?;
        }
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
