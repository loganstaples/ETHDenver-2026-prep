//! Status command implementation
//!
//! Provides comprehensive status information about the node, network, and training progress.
//! Supports both one-time queries and continuous watch mode.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::progress::ProgressDisplay;

/// Status command options
#[derive(Debug, Clone)]
pub struct StatusOptions {
    /// Show detailed status information
    pub detailed: bool,
    /// Watch mode - continuously update
    pub watch: bool,
    /// Specific model ID to check
    pub model_id: Option<u64>,
    /// Refresh interval in seconds (for watch mode)
    pub interval: u64,
    /// Output format (text, json)
    pub format: OutputFormat,
    /// Show specific sections only
    pub sections: Vec<StatusSection>,
}

impl Default for StatusOptions {
    fn default() -> Self {
        Self {
            detailed: false,
            watch: false,
            model_id: None,
            interval: 5,
            format: OutputFormat::Text,
            sections: vec![],
        }
    }
}

/// Output format for status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Compact,
}

/// Status sections to display
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusSection {
    Node,
    Network,
    Training,
    Staking,
    Proofs,
    Peers,
    All,
}

/// Complete status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatusInfo {
    /// Node status
    pub node: NodeStatus,
    /// Network status
    pub network: NetworkConnectionStatus,
    /// Training status
    pub training: TrainingStatus,
    /// Staking status
    pub staking: StakingStatus,
    /// Proof status
    pub proofs: ProofStatus,
    /// Connected peers
    pub peers: Vec<PeerStatus>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node ID
    pub node_id: String,
    /// Node name
    pub name: String,
    /// Node status (online, offline, syncing)
    pub status: String,
    /// Node version
    pub version: String,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage in MB
    pub memory_mb: u32,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Disk usage in MB
    pub disk_mb: u64,
    /// Capabilities
    pub capabilities: Vec<String>,
}

/// Network connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnectionStatus {
    /// Is connected to network
    pub connected: bool,
    /// Number of connected peers
    pub peer_count: u32,
    /// RPC endpoint
    pub rpc_url: String,
    /// Current block height
    pub block_height: u64,
    /// Chain ID
    pub chain_id: u64,
    /// Network latency in ms
    pub latency_ms: u32,
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
}

/// Training progress status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    /// Is training active
    pub is_active: bool,
    /// Model ID being trained
    pub model_id: Option<u64>,
    /// Current round
    pub current_round: u64,
    /// Total rounds
    pub total_rounds: u64,
    /// Round progress (0-100)
    pub round_progress: f32,
    /// Current loss
    pub current_loss: Option<f64>,
    /// Loss history (last N rounds)
    pub loss_history: Vec<f64>,
    /// Current error bound
    pub error_bound: f64,
    /// Maximum error bound
    pub max_error: f64,
    /// Average round time in ms
    pub avg_round_time_ms: u64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
}

/// Staking status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingStatus {
    /// Total staked amount in ETH
    pub staked_amount: f64,
    /// Is stake locked
    pub is_locked: bool,
    /// Lock remaining time
    pub lock_remaining_seconds: Option<u64>,
    /// Has been slashed
    pub is_slashed: bool,
    /// Slashed amount
    pub slashed_amount: f64,
    /// Reputation score
    pub reputation: f64,
    /// Pending rewards in ETH
    pub pending_rewards: f64,
}

/// Proof generation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatus {
    /// Proofs generated
    pub generated: u64,
    /// Proofs submitted
    pub submitted: u64,
    /// Proofs verified
    pub verified: u64,
    /// Proofs failed
    pub failed: u64,
    /// Average proof time in ms
    pub avg_proof_time_ms: u64,
    /// Proof success rate
    pub success_rate: f64,
    /// Pending proofs
    pub pending: u32,
}

/// Individual peer status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStatus {
    /// Peer ID
    pub id: String,
    /// Peer address
    pub address: String,
    /// Peer role
    pub role: String,
    /// Connection status
    pub status: String,
    /// Latency in ms
    pub latency_ms: u32,
    /// Reputation score
    pub reputation: f64,
    /// Last seen timestamp
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

/// Status command handler
pub struct StatusCommand {
    /// Progress display
    progress: ProgressDisplay,
    /// Cached status
    cached_status: Arc<RwLock<Option<NetworkStatusInfo>>>,
    /// Last update time
    last_update: Arc<RwLock<Option<Instant>>>,
}

impl StatusCommand {
    /// Create a new status command
    pub fn new() -> Self {
        Self {
            progress: ProgressDisplay::new(),
            cached_status: Arc::new(RwLock::new(None)),
            last_update: Arc::new(RwLock::new(None)),
        }
    }

    /// Execute the status command
    pub async fn execute(&self, options: StatusOptions) -> Result<NetworkStatusInfo> {
        // Fetch current status
        let status = self.fetch_status(&options).await?;

        // Cache the status
        *self.cached_status.write().await = Some(status.clone());
        *self.last_update.write().await = Some(Instant::now());

        Ok(status)
    }

    /// Execute in watch mode
    pub async fn watch(&self, options: StatusOptions, mut shutdown: tokio::sync::broadcast::Receiver<()>) -> Result<()> {
        loop {
            // Clear screen
            print!("\x1B[2J\x1B[1;1H");

            // Fetch and display status
            let status = self.fetch_status(&options).await?;
            self.display(&status, &options);

            println!("\n{}", format!("Refreshing every {}s... (Ctrl+C to stop)", options.interval).dimmed());

            // Wait for interval or shutdown
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(options.interval)) => {}
                _ = shutdown.recv() => {
                    println!("\nStopping status watch...");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Fetch current status
    async fn fetch_status(&self, options: &StatusOptions) -> Result<NetworkStatusInfo> {
        // Simulated status data (in production, fetch from actual sources)
        let node = NodeStatus {
            node_id: "helix-node-a1b2c3d4".to_string(),
            name: "helix-demo-node".to_string(),
            status: "online".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: 9252,
            memory_mb: 1234,
            cpu_percent: 23.5,
            disk_mb: 5678,
            capabilities: vec!["train".to_string(), "prove".to_string()],
        };

        let network = NetworkConnectionStatus {
            connected: true,
            peer_count: 4,
            rpc_url: "http://localhost:8545".to_string(),
            block_height: 12_345_678,
            chain_id: 31337,
            latency_ms: 15,
            messages_sent: 15678,
            messages_received: 14532,
        };

        let training = TrainingStatus {
            is_active: true,
            model_id: options.model_id.or(Some(0)),
            current_round: 42,
            total_rounds: 100,
            round_progress: 67.0,
            current_loss: Some(0.234),
            loss_history: vec![0.892, 0.654, 0.478, 0.356, 0.298, 0.267, 0.245, 0.234],
            error_bound: 45.2,
            max_error: 1000.0,
            avg_round_time_ms: 2500,
            eta_seconds: Some(145000),
        };

        let staking = StakingStatus {
            staked_amount: 1.5,
            is_locked: true,
            lock_remaining_seconds: Some(432000),
            is_slashed: false,
            slashed_amount: 0.0,
            reputation: 0.98,
            pending_rewards: 0.023,
        };

        let proofs = ProofStatus {
            generated: 127,
            submitted: 127,
            verified: 125,
            failed: 2,
            avg_proof_time_ms: 1200,
            success_rate: 98.4,
            pending: 0,
        };

        let peers = vec![
            PeerStatus {
                id: "helix-node-e5f6g7h8".to_string(),
                address: "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199".to_string(),
                role: "Worker".to_string(),
                status: "active".to_string(),
                latency_ms: 15,
                reputation: 0.95,
                last_seen: chrono::Utc::now(),
            },
            PeerStatus {
                id: "helix-node-i9j0k1l2".to_string(),
                address: "0xdD2FD4581271e230360230F9337D5c0430Bf44C0".to_string(),
                role: "Worker".to_string(),
                status: "active".to_string(),
                latency_ms: 23,
                reputation: 0.92,
                last_seen: chrono::Utc::now(),
            },
            PeerStatus {
                id: "helix-agg-01".to_string(),
                address: "0x5FbDB2315678afecb367f032d93F642f64180aa3".to_string(),
                role: "Aggregator".to_string(),
                status: "active".to_string(),
                latency_ms: 8,
                reputation: 1.0,
                last_seen: chrono::Utc::now(),
            },
            PeerStatus {
                id: "helix-node-m3n4o5p6".to_string(),
                address: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512".to_string(),
                role: "Worker".to_string(),
                status: "idle".to_string(),
                latency_ms: 45,
                reputation: 0.88,
                last_seen: chrono::Utc::now() - chrono::Duration::seconds(30),
            },
        ];

        Ok(NetworkStatusInfo {
            node,
            network,
            training,
            staking,
            proofs,
            peers,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Display status
    pub fn display(&self, status: &NetworkStatusInfo, options: &StatusOptions) {
        match options.format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(status).unwrap_or_default());
                return;
            }
            OutputFormat::Compact => {
                self.display_compact(status);
                return;
            }
            OutputFormat::Text => {}
        }

        println!("{}", "═".repeat(60).cyan());
        println!("{}", " HELIX Network Status".cyan().bold());
        println!("{}", "═".repeat(60).cyan());

        // Node Status
        println!("\n{}", "Node Status:".yellow().bold());
        let status_icon = if status.node.status == "online" { "●".green() } else { "●".red() };
        println!("  Node ID:     {}", status.node.node_id);
        println!("  Status:      {} {}", status_icon, status.node.status.to_uppercase());
        println!("  Version:     {}", status.node.version);
        println!("  Uptime:      {}", format_duration(status.node.uptime_seconds));
        if options.detailed {
            println!("  Memory:      {} MB", status.node.memory_mb);
            println!("  CPU:         {:.1}%", status.node.cpu_percent);
            println!("  Disk:        {} MB", status.node.disk_mb);
        }

        // Network Status
        println!("\n{}", "Network Status:".yellow().bold());
        let conn_icon = if status.network.connected { "●".green() } else { "●".red() };
        println!("  Connected:   {} {}", conn_icon, if status.network.connected { "Yes" } else { "No" });
        println!("  Peers:       {} connected", status.network.peer_count);
        println!("  Block:       {}", status.network.block_height);
        println!("  Latency:     {} ms", status.network.latency_ms);
        if options.detailed {
            println!("  RPC:         {}", status.network.rpc_url);
            println!("  Chain ID:    {}", status.network.chain_id);
            println!("  Msgs Sent:   {}", status.network.messages_sent);
            println!("  Msgs Recv:   {}", status.network.messages_received);
        }

        // Training Status
        println!("\n{}", "Training Status:".yellow().bold());
        if status.training.is_active {
            println!("  Model ID:    {}", status.training.model_id.unwrap_or(0));
            println!("  Round:       {}/{} ({:.1}%)",
                status.training.current_round,
                status.training.total_rounds,
                (status.training.current_round as f32 / status.training.total_rounds as f32) * 100.0
            );

            // Progress bar
            let progress = (status.training.round_progress / 100.0 * 30.0) as usize;
            let bar = format!("[{}{}] {:.1}%",
                "█".repeat(progress).cyan(),
                "░".repeat(30 - progress),
                status.training.round_progress
            );
            println!("  Progress:    {}", bar);

            if let Some(loss) = status.training.current_loss {
                println!("  Loss:        {:.6}", loss);
            }
            println!("  Error Bound: {:.1} / {:.0} max", status.training.error_bound, status.training.max_error);

            if let Some(eta) = status.training.eta_seconds {
                println!("  ETA:         {}", format_duration(eta));
            }

            if options.detailed && !status.training.loss_history.is_empty() {
                println!("\n  {}", "Loss History:".dimmed());
                let min_loss = status.training.loss_history.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_loss = status.training.loss_history.iter().cloned().fold(0.0, f64::max);
                let range = max_loss - min_loss;

                print!("  ");
                for loss in &status.training.loss_history {
                    let normalized = if range > 0.0 { (loss - min_loss) / range } else { 0.5 };
                    let bar_height = (normalized * 5.0) as usize;
                    let char = match bar_height {
                        0 => "▁",
                        1 => "▂",
                        2 => "▃",
                        3 => "▄",
                        4 => "▅",
                        _ => "▆",
                    };
                    print!("{}", char.green());
                }
                println!(" ({:.4} → {:.4})", status.training.loss_history.first().unwrap_or(&0.0),
                    status.training.loss_history.last().unwrap_or(&0.0));
            }
        } else {
            println!("  Status:      {} No active training", "●".yellow());
        }

        // Staking Status
        println!("\n{}", "Staking:".yellow().bold());
        println!("  Staked:      {} ETH", status.staking.staked_amount);
        if status.staking.is_locked {
            if let Some(remaining) = status.staking.lock_remaining_seconds {
                println!("  Lock:        {} {} remaining", "●".yellow(), format_duration(remaining));
            }
        }
        println!("  Reputation:  {:.0}%", status.staking.reputation * 100.0);
        if status.staking.is_slashed {
            println!("  {} Slashed: {} ETH", "⚠".red(), status.staking.slashed_amount);
        }
        if status.staking.pending_rewards > 0.0 {
            println!("  Rewards:     {} ETH (pending)", status.staking.pending_rewards);
        }

        // Proof Status
        if options.detailed {
            println!("\n{}", "Proofs:".yellow().bold());
            println!("  Generated:   {}", status.proofs.generated);
            println!("  Verified:    {}", status.proofs.verified);
            println!("  Failed:      {}", status.proofs.failed);
            println!("  Success:     {:.1}%", status.proofs.success_rate);
            println!("  Avg Time:    {} ms", status.proofs.avg_proof_time_ms);
        }

        // Peers
        if options.detailed {
            println!("\n{}", "Connected Peers:".yellow().bold());
            println!("  ┌─────────────────┬──────────┬────────────┬──────────┐");
            println!("  │ Peer ID         │ Status   │ Role       │ Latency  │");
            println!("  ├─────────────────┼──────────┼────────────┼──────────┤");
            for peer in &status.peers {
                let status_icon = if peer.status == "active" { "●".green() } else { "●".yellow() };
                println!(
                    "  │ {:15} │ {} {:6} │ {:10} │ {:>6}ms │",
                    &peer.id[..15.min(peer.id.len())],
                    status_icon,
                    peer.status,
                    peer.role,
                    peer.latency_ms
                );
            }
            println!("  └─────────────────┴──────────┴────────────┴──────────┘");
        }

        println!("\n{}", "═".repeat(60).cyan());
        println!("  Last updated: {}", status.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    }

    /// Display compact status
    fn display_compact(&self, status: &NetworkStatusInfo) {
        let status_icon = if status.node.status == "online" { "●".green() } else { "●".red() };
        let training_icon = if status.training.is_active { "●".blue() } else { "●".dimmed() };

        println!(
            "{} {} | {} peers | Round {}/{} {} | Loss: {:.4} | Proofs: {}/{}",
            status_icon,
            status.node.name,
            status.network.peer_count,
            status.training.current_round,
            status.training.total_rounds,
            training_icon,
            status.training.current_loss.unwrap_or(0.0),
            status.proofs.verified,
            status.proofs.generated
        );
    }

    /// Get cached status
    pub async fn get_cached(&self) -> Option<NetworkStatusInfo> {
        self.cached_status.read().await.clone()
    }
}

impl Default for StatusCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Format seconds as human-readable duration
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else if seconds < 86400 {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    } else {
        format!("{}d {}h", seconds / 86400, (seconds % 86400) / 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m");
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_status_options_default() {
        let options = StatusOptions::default();
        assert!(!options.detailed);
        assert!(!options.watch);
        assert_eq!(options.interval, 5);
    }

    #[tokio::test]
    async fn test_status_command() {
        let cmd = StatusCommand::new();
        let options = StatusOptions::default();
        let status = cmd.execute(options).await.unwrap();

        assert!(!status.node.node_id.is_empty());
        assert!(status.network.connected);
    }
}
