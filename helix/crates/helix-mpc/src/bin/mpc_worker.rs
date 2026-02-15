//! Standalone MPC worker binary for multi-process TCP training.
//!
//! Each worker is an independent process that:
//! 1. Binds to a local TCP port
//! 2. Connects to all peer workers to form a full mesh
//! 3. Participates in the MPC training protocol
//! 4. Reports results (losses, checkpoints, final weight shares)
//!
//! Usage:
//!   # Start 3 workers on different ports:
//!   cargo run -p helix-mpc --features network-mpc --bin mpc-worker -- \
//!     --party 0 --bind 127.0.0.1:9000 \
//!     --peer 1=127.0.0.1:9001 --peer 2=127.0.0.1:9002 \
//!     --config training_config.json
//!
//!   cargo run -p helix-mpc --features network-mpc --bin mpc-worker -- \
//!     --party 1 --bind 127.0.0.1:9001 \
//!     --peer 0=127.0.0.1:9000 --peer 2=127.0.0.1:9002 \
//!     --config training_config.json
//!
//!   cargo run -p helix-mpc --features network-mpc --bin mpc-worker -- \
//!     --party 2 --bind 127.0.0.1:9002 \
//!     --peer 0=127.0.0.1:9000 --peer 1=127.0.0.1:9001 \
//!     --config training_config.json

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[cfg(feature = "network-mpc")]
use helix_mpc::session::transport::TcpTransport;

use helix_mpc::e2e_integration::MPCIntegrationConfig;
use helix_mpc::field::Fr;
use helix_mpc::mac_verification::MACVerificationConfig;
use helix_mpc::mpc_trainer::{MPCTrainer, MPCTrainerConfig, ModelWeights};
use helix_mpc::security::commitment::PedersenGenerators;
use helix_mpc::share_distribution::{CheckpointCommitment, WeightShare};
use helix_mpc::types::PartyId;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// HELIX MPC Worker — standalone process for multi-party computation training.
///
/// Each worker binds to a TCP port, connects to peers, and participates in
/// the MPC training protocol (secret-shared weights, Beaver triple multiplication,
/// SPDZ MAC verification, Pedersen commitment checkpoints).
#[derive(Parser, Debug)]
#[command(name = "mpc-worker", version, about)]
struct Args {
    /// This worker's party index (0-based).
    #[arg(short, long)]
    party: usize,

    /// TCP address to bind to (e.g., 127.0.0.1:9000).
    #[arg(short, long)]
    bind: SocketAddr,

    /// Peer addresses in the format INDEX=HOST:PORT.
    /// Repeat for each peer (e.g., --peer 1=127.0.0.1:9001 --peer 2=127.0.0.1:9002).
    #[arg(long, value_parser = parse_peer)]
    peer: Vec<(usize, SocketAddr)>,

    /// Path to a JSON config file with training parameters.
    /// If not specified, uses default small-model configuration.
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Number of training steps (overrides config file).
    #[arg(short = 'n', long)]
    steps: Option<usize>,

    /// Learning rate (overrides config file).
    #[arg(long)]
    lr: Option<f64>,

    /// MAC check interval (0 to disable, overrides config file).
    #[arg(long)]
    mac_interval: Option<u64>,

    /// Pedersen checkpoint interval (overrides config file).
    #[arg(long)]
    checkpoint_interval: Option<usize>,

    /// Beaver triple batch size (overrides config file).
    #[arg(long)]
    beaver_batch: Option<usize>,

    /// Random seed for deterministic execution.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Verbose logging output.
    #[arg(short, long)]
    verbose: bool,

    /// Output results to JSON file.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

/// Parses a peer specification "INDEX=HOST:PORT".
fn parse_peer(s: &str) -> Result<(usize, SocketAddr), String> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid peer format '{}', expected INDEX=HOST:PORT", s));
    }
    let index: usize = parts[0]
        .parse()
        .map_err(|e| format!("Invalid peer index '{}': {}", parts[0], e))?;
    let addr: SocketAddr = parts[1]
        .parse()
        .map_err(|e| format!("Invalid peer address '{}': {}", parts[1], e))?;
    Ok((index, addr))
}

/// Worker result serialized to JSON.
#[derive(serde::Serialize)]
struct WorkerResult {
    party_index: usize,
    steps_completed: usize,
    losses: Vec<f64>,
    final_loss: f64,
    mac_checks_passed: usize,
    cheater_detected: bool,
    checkpoints: Vec<CheckpointRecord>,
    training_time_ms: u128,
}

#[derive(serde::Serialize)]
struct CheckpointRecord {
    step: usize,
    loss: f64,
}

#[cfg(not(feature = "network-mpc"))]
fn main() {
    eprintln!("ERROR: mpc-worker requires the 'network-mpc' feature.");
    eprintln!("Build with: cargo build -p helix-mpc --features network-mpc --bin mpc-worker");
    std::process::exit(1);
}

#[cfg(feature = "network-mpc")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Initialize tracing.
    let filter = if args.verbose {
        "helix_mpc=debug,mpc_worker=debug"
    } else {
        "helix_mpc=info,mpc_worker=info"
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(filter)
        }))
        .with_target(false)
        .init();

    info!(
        party = args.party,
        bind = %args.bind,
        peers = args.peer.len(),
        "Starting MPC worker"
    );

    // Load or build training config.
    let integration_config = if let Some(config_path) = &args.config {
        let data = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;
        serde_json::from_str::<MPCIntegrationConfig>(&data)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?
    } else {
        MPCIntegrationConfig::default()
    };

    // Apply CLI overrides.
    let num_steps = args.steps.unwrap_or(integration_config.num_steps);
    let learning_rate = args.lr.unwrap_or(integration_config.learning_rate);
    let mac_check_interval = args.mac_interval.unwrap_or(integration_config.mac_check_interval);
    let checkpoint_interval = args.checkpoint_interval.unwrap_or(integration_config.checkpoint_interval);
    let beaver_batch_size = args.beaver_batch.unwrap_or(integration_config.beaver_batch_size);

    let trainer_config = MPCTrainerConfig {
        d_in: integration_config.d_in,
        d_hid: integration_config.d_hid,
        d_out: integration_config.d_out,
        learning_rate,
        num_parties: args.peer.len() + 1, // peers + self
        reshare_interval: 0,
        beaver_batch_size,
        generate_proofs: false,
        base_error: 1e-6,
        checkpoint_interval: 1,
        mac_config: if mac_check_interval > 0 {
            Some(MACVerificationConfig {
                check_interval: mac_check_interval,
                enable_cheater_identification: true,
                mac_seed: args.seed.wrapping_mul(0xCAFE_BABE),
            })
        } else {
            None
        },
    };

    let initial_weights = if args.party == 0 {
        integration_config.initial_weights.as_ref().map(|iw| {
            ModelWeights::from_f64(&iw.w1, &iw.b1, &iw.w2, &iw.b2)
        })
    } else {
        None
    };

    let training_data = integration_config.training_data.clone();

    // Build peer address map.
    let mut peer_addrs: HashMap<PartyId, SocketAddr> = HashMap::new();
    for (idx, addr) in &args.peer {
        peer_addrs.insert(PartyId::from_index(*idx), *addr);
    }

    info!(
        party = args.party,
        bind = %args.bind,
        peers = ?peer_addrs.keys().collect::<Vec<_>>(),
        d_in = trainer_config.d_in,
        d_hid = trainer_config.d_hid,
        d_out = trainer_config.d_out,
        num_steps = num_steps,
        "Establishing TCP transport mesh"
    );

    // Establish TCP transport.
    let start = std::time::Instant::now();
    let party_id = PartyId::from_index(args.party);
    let transport = TcpTransport::bind(args.bind, party_id, &peer_addrs)
        .await
        .map_err(|e| anyhow::anyhow!("TcpTransport::bind failed: {}", e))?;

    let mesh_time = start.elapsed();
    info!(
        party = args.party,
        mesh_time_ms = mesh_time.as_millis(),
        "TCP mesh established"
    );

    // Create trainer and share weights.
    let mut trainer = MPCTrainer::new(trainer_config.clone(), transport, args.party, args.seed);
    trainer.share_weights(initial_weights).await?;
    info!(party = args.party, "Weight shares initialized");

    // Generate Beaver triples.
    let triples_per_step = (trainer_config.d_hid + trainer_config.d_out * trainer_config.d_hid + trainer_config.d_hid) * 2 + 32;
    let total_triples = triples_per_step * num_steps;
    let batch_size = beaver_batch_size.max(total_triples);
    trainer.generate_beaver_triples(batch_size).await?;
    info!(
        party = args.party,
        triples = trainer.beaver_triples_remaining(),
        "Beaver triples generated"
    );

    // Training loop.
    let use_mac = trainer_config.mac_config.is_some();
    let mut losses = Vec::with_capacity(num_steps);
    let mut mac_checks_passed = 0usize;
    let mut cheater_detected = false;
    let mut steps_completed = 0usize;

    // Checkpoint state.
    let generators = PedersenGenerators::default();
    let checkpoint_system = CheckpointCommitment::with_generators(generators);
    let mut checkpoints: Vec<CheckpointRecord> = Vec::new();
    let mut rng = ChaCha20Rng::seed_from_u64(args.seed.wrapping_add(args.party as u64 * 1000));

    let training_start = std::time::Instant::now();

    for step in 0..num_steps {
        let data_idx = step % training_data.len().max(1);
        let (input, target) = if !training_data.is_empty() {
            (&training_data[data_idx].0, &training_data[data_idx].1)
        } else {
            // Default dummy data if none provided.
            (&vec![0.5; trainer_config.d_in], &vec![0.5; trainer_config.d_out])
        };

        let step_result = if use_mac {
            use helix_mpc::error::MPCError;
            match trainer.training_step_with_mac(input, target).await {
                Ok(result) => {
                    if trainer_config.mac_config.as_ref().map_or(false, |mc| {
                        mc.check_interval > 0 && (step as u64 + 1) % mc.check_interval == 0
                    }) {
                        mac_checks_passed += 1;
                    }
                    result
                }
                Err(MPCError::MACCheckFailed { step: fail_step, cheater }) => {
                    error!(
                        party = args.party,
                        step = fail_step,
                        cheater = ?cheater,
                        "MAC check failed - cheater detected!"
                    );
                    cheater_detected = true;
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        } else {
            trainer.training_step_unproved(input, target).await?
        };

        losses.push(step_result.loss);
        steps_completed += 1;

        if step % 10 == 0 || step == num_steps - 1 {
            info!(
                party = args.party,
                step = step,
                loss = format!("{:.6}", step_result.loss),
                "Training step"
            );
        }

        // Pedersen checkpoint.
        if checkpoint_interval > 0 && (step + 1) % checkpoint_interval == 0 {
            let (w1, b1, w2, b2) = trainer.weight_shares();
            let all_weights: Vec<Fr> = w1.iter()
                .chain(b1.iter())
                .chain(w2.iter())
                .chain(b2.iter())
                .cloned()
                .collect();

            let blindings: Vec<Fr> = (0..all_weights.len())
                .map(|_| Fr::random(&mut rng))
                .collect();

            let weight_share = WeightShare {
                party: PartyId::from_index(args.party),
                index: args.party,
                data: all_weights,
                shape: vec![w1.len() + b1.len() + w2.len() + b2.len()],
            };

            if let Ok(_commitment_share) = checkpoint_system.compute_share(&weight_share, &blindings) {
                checkpoints.push(CheckpointRecord {
                    step: step + 1,
                    loss: step_result.loss,
                });
                info!(
                    party = args.party,
                    step = step + 1,
                    "Pedersen checkpoint computed"
                );
            }
        }
    }

    let training_time_ms = training_start.elapsed().as_millis();
    let final_loss = losses.last().copied().unwrap_or(0.0);

    info!(
        party = args.party,
        steps = steps_completed,
        final_loss = format!("{:.6}", final_loss),
        mac_checks = mac_checks_passed,
        checkpoints = checkpoints.len(),
        cheater = cheater_detected,
        time_ms = training_time_ms,
        "Worker training complete"
    );

    // Build result.
    let result = WorkerResult {
        party_index: args.party,
        steps_completed,
        losses,
        final_loss,
        mac_checks_passed,
        cheater_detected,
        checkpoints,
        training_time_ms,
    };

    // Output results.
    if let Some(output_path) = &args.output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(output_path, json)?;
        info!(party = args.party, path = %output_path.display(), "Results written to file");
    } else {
        // Print summary to stdout.
        println!("=== Worker {} Results ===", args.party);
        println!("Steps completed: {}", result.steps_completed);
        println!("Final loss: {:.6}", result.final_loss);
        println!("MAC checks passed: {}", result.mac_checks_passed);
        println!("Cheater detected: {}", result.cheater_detected);
        println!("Checkpoints: {}", result.checkpoints.len());
        println!("Training time: {}ms", result.training_time_ms);
    }

    Ok(())
}
