//! HELIX Node — Configuration-driven P2P training node.
//!
//! Usage:
//!   cargo run -p helix-node -- --config node.toml
//!   cargo run -p helix-node -- --config config.json
//!   cargo run -p helix-node -- --config node.toml --verbose
//!
//! The node reads all settings from the config file (TOML or JSON) and boots
//! all subsystems in dependency order via `NodeRuntime`.
//!
//! Environment variables:
//!   HELIX_PRIVATE_KEY      — Hex private key for on-chain transactions (never in config)
//!   HELIX_MPC_STORAGE_KEY  — Password for encrypted MPC weight storage
//!   RUST_LOG               — Logging filter (default: info)

use std::process;
use std::time::Duration;

use helix_node::config::NodeConfig;
use helix_node::runtime::{shutdown_signal, NodeRuntime};
use log::{error, info};

fn print_usage() {
    eprintln!(
        "Usage: helix-node --config <path>\n\
         \n\
         Options:\n\
         \x20 --config <path>   Path to TOML or JSON configuration file (required)\n\
         \x20 --validate        Validate config and exit without starting\n\
         \x20 --verbose         Enable debug-level logging\n\
         \x20 --help            Show this help message"
    );
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        return Err("No arguments provided. Use --help for usage.".into());
    }

    let mut config_path = None;
    let mut validate_only = false;
    let mut verbose = false;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                if i >= args.len() {
                    return Err("--config requires a file path argument".into());
                }
                config_path = Some(args[i].clone());
            }
            "--validate" => {
                validate_only = true;
            }
            "--verbose" | "-v" => {
                verbose = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            other => {
                return Err(format!("Unknown argument: {}. Use --help for usage.", other));
            }
        }
        i += 1;
    }

    let config_path = config_path.ok_or("--config <path> is required")?;
    Ok(CliArgs {
        config_path,
        validate_only,
        verbose,
    })
}

struct CliArgs {
    config_path: String,
    validate_only: bool,
    verbose: bool,
}

#[tokio::main]
async fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("Error: {}", msg);
            print_usage();
            process::exit(1);
        }
    };

    // Initialize logging
    let default_filter = if args.verbose { "debug" } else { "info" };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", default_filter);
    }
    env_logger::init();

    // Load and validate config
    info!("Loading configuration from {}", args.config_path);
    let config = match NodeConfig::from_file(&args.config_path) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to load config: {}", e);
            process::exit(1);
        }
    };

    if args.validate_only {
        println!("Configuration is valid.");
        println!(
            "  Role:          {}",
            config.role_str()
        );
        println!("  Listen:        {}", config.listen_addr);
        println!("  RPC port:      {}", config.rpc_port);
        println!("  HTTP port:     {}", config.http_port);
        println!(
            "  Model:         {}x{}x{}",
            config.training.d_in, config.training.d_hid, config.training.d_out
        );
        println!("  MPC:           {}", if config.mpc.enabled { "enabled" } else { "disabled" });
        println!(
            "  Chain:         {}",
            if config.chain.is_some() { "configured" } else { "none" }
        );
        process::exit(0);
    }

    // Print startup banner
    info!("=== HELIX Node {} ===", env!("CARGO_PKG_VERSION"));
    info!(
        "Role: {}, Listen: {}, RPC: {}, HTTP: {}",
        config.role_str(),
        config.listen_addr,
        config.rpc_port,
        config.http_port,
    );
    info!(
        "Model: {}x{}x{} (lr={}, seed={})",
        config.training.d_in,
        config.training.d_hid,
        config.training.d_out,
        config.training.learning_rate,
        config.training.model_seed,
    );
    if config.mpc.enabled {
        info!(
            "MPC: {} parties, reshare every {} steps",
            config.mpc.num_parties, config.mpc.reshare_interval,
        );
    }
    if let Some(ref chain) = config.chain {
        info!(
            "Chain: coordinator={}, rpc={}",
            chain.coordinator_address, config.rpc_url,
        );
    }

    // Create and run the node runtime
    let runtime = NodeRuntime::new(config);
    let shutdown_timeout = Duration::from_secs(
        runtime.config().fault_tolerance.shutdown_timeout_secs,
    );

    // Run with signal handling
    tokio::select! {
        result = runtime.run() => {
            match result {
                Ok(()) => info!("Node exited cleanly"),
                Err(e) => {
                    error!("Node exited with error: {}", e);
                    process::exit(1);
                }
            }
        }
        _ = shutdown_signal() => {
            info!("Shutdown signal received, stopping node...");
            runtime.shutdown();

            // Give subsystems time to wind down
            tokio::time::sleep(shutdown_timeout).await;
            info!("Shutdown complete");
        }
    }
}
