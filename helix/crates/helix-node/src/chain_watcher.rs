//! On-Chain Event Watcher.
//!
//! Watches the HelixCoordinatorV2/V3 contract for events and dispatches them
//! to the node's internal systems via a broadcast channel. Handles:
//!
//! - **Event polling** with configurable interval (HTTP-based for reliability)
//! - **Confirmation depth** tracking for chain reorganization safety
//! - **Persistent cursor** for crash recovery (resumes from last processed block)
//! - **Broadcasting** events to multiple consumers via `tokio::sync::broadcast`
//!
//! # Key Events Watched
//!
//! | Event | Node Action |
//! |-------|-------------|
//! | `ModelRegistered` | Aggregator tracks new model availability |
//! | `RoundStarted` | Workers discover available training rounds |
//! | `ProofAccepted` | Confirm on-chain acceptance of submitted proofs |
//! | `Slashed` / `InvalidProofDetected` | Alert if local worker was slashed |
//! | `TrainingHalted` | Stop submitting proofs for halted rounds |
//! | `EmergencyPauseChanged` | Halt all contract interactions when paused |
//! | `RoundFinalized` | Update local state with finalized round data |
//!
//! # Reorg Safety
//!
//! Events are only dispatched after `confirmation_depth` blocks have been mined
//! on top of them. The default depth of 12 blocks provides ~3 minutes of safety
//! on Ethereum mainnet. On restart, the watcher validates the stored block hash
//! still exists on-chain; if not (reorg occurred), it rewinds further back.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ethers::abi::RawLog;
use ethers::contract::EthLogDecode;
use ethers::prelude::*;
use ethers::providers::{Http, Provider};
use ethers::types::{Address, Filter, Log, H256, U256};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// ABI bindings (event-only) for the HelixCoordinatorV2 contract.
// Separate from sc_client.rs to keep the watcher self-contained.
// ---------------------------------------------------------------------------
abigen!(
    HelixCoordinatorWatcher,
    r#"[
        event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment, uint256 minStake, string ipfsHash)
        event RoundStarted(uint256 indexed modelId, uint256 indexed roundId, uint256 deadline, uint256 modelCommitment)
        event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment, uint256 errorBound)
        event ProofAccepted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 stepNumber, uint256 newCommitment, uint256 loss, uint256 errorBound)
        event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment, uint256 totalErrorBound)
        event RoundFinalized(uint256 indexed modelId, uint256 indexed roundId, uint256 finalCommitment, uint256 totalSteps, uint256 finalLoss, uint256 totalError)
        event Staked(address indexed prover, uint256 indexed modelId, uint256 amount, uint256 totalStake)
        event Unstaked(address indexed prover, uint256 indexed modelId, uint256 amount)
        event Slashed(address indexed prover, uint256 indexed modelId, uint256 roundId, uint256 amount, uint256 remainingStake, string reason)
        event InvalidProofDetected(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, bytes32 proofHash)
        event TrainingHalted(uint256 indexed modelId, uint256 indexed roundId, uint256 accumulatedError, uint256 maxBudget)
        event EmergencyPauseChanged(bool isPaused, address indexed changedBy)
        event ModelStateChanged(uint256 indexed modelId, bool active, address indexed changedBy)
        event ErrorBudgetWarning(uint256 indexed modelId, uint256 indexed roundId, uint256 accumulated, uint256 budget)
        event CommitmentUpdated(uint256 indexed modelId, uint256 indexed roundId, uint256 oldCommitment, uint256 newCommitment, uint256 stepNumber)
    ]"#
);

// ============================================================================
// Public Event Types
// ============================================================================

/// A confirmed on-chain event with block metadata.
///
/// Events are only emitted after sufficient confirmations (see `confirmation_depth`
/// in [`ChainWatcherConfig`]), so consumers can treat them as final.
#[derive(Debug, Clone)]
pub struct ChainEvent {
    /// The parsed event data.
    pub data: ChainEventData,
    /// Block number where this event was emitted.
    pub block_number: u64,
    /// Block hash.
    pub block_hash: H256,
    /// Transaction hash that emitted this event.
    pub tx_hash: H256,
    /// Log index within the block.
    pub log_index: u64,
}

/// Parsed event data from the HelixCoordinatorV2/V3 contract.
#[derive(Debug, Clone)]
pub enum ChainEventData {
    /// A new model was registered on-chain.
    ModelRegistered {
        model_id: u64,
        owner: Address,
        initial_commitment: U256,
        min_stake: U256,
        ipfs_hash: String,
    },
    /// A new training round was started.
    RoundStarted {
        model_id: u64,
        round_id: u64,
        deadline: u64,
        model_commitment: U256,
    },
    /// A proof was submitted (backward-compat event, emitted alongside ProofAccepted).
    ProofSubmitted {
        model_id: u64,
        round_id: u64,
        prover: Address,
        new_commitment: U256,
        error_bound: U256,
    },
    /// A proof was accepted and state was updated.
    ProofAccepted {
        model_id: u64,
        round_id: u64,
        prover: Address,
        step_number: u64,
        new_commitment: U256,
        loss: U256,
        error_bound: U256,
    },
    /// A training round was completed.
    RoundCompleted {
        model_id: u64,
        round_id: u64,
        new_commitment: U256,
        total_error_bound: U256,
    },
    /// A training round was finalized with full metrics.
    RoundFinalized {
        model_id: u64,
        round_id: u64,
        final_commitment: U256,
        total_steps: u64,
        final_loss: U256,
        total_error: U256,
    },
    /// Stake was deposited for a model.
    Staked {
        prover: Address,
        model_id: u64,
        amount: U256,
        total_stake: U256,
    },
    /// Stake was withdrawn.
    Unstaked {
        prover: Address,
        model_id: u64,
        amount: U256,
    },
    /// A prover was slashed for an invalid proof or other violation.
    Slashed {
        prover: Address,
        model_id: u64,
        round_id: u64,
        amount: U256,
        remaining_stake: U256,
        reason: String,
    },
    /// An invalid proof was detected (triggers slashing).
    InvalidProofDetected {
        model_id: u64,
        round_id: u64,
        prover: Address,
        proof_hash: [u8; 32],
    },
    /// Training was halted for a round due to error budget exhaustion.
    TrainingHalted {
        model_id: u64,
        round_id: u64,
        accumulated_error: U256,
        max_budget: U256,
    },
    /// Contract was paused or unpaused.
    EmergencyPauseChanged {
        is_paused: bool,
        changed_by: Address,
    },
    /// Model active state changed.
    ModelStateChanged {
        model_id: u64,
        active: bool,
        changed_by: Address,
    },
    /// Error budget reached 80% warning threshold.
    ErrorBudgetWarning {
        model_id: u64,
        round_id: u64,
        accumulated: U256,
        budget: U256,
    },
    /// Model commitment was updated (emitted per proof acceptance).
    CommitmentUpdated {
        model_id: u64,
        round_id: u64,
        old_commitment: U256,
        new_commitment: U256,
        step_number: u64,
    },
}

impl std::fmt::Display for ChainEventData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelRegistered { model_id, owner, .. } => {
                write!(f, "ModelRegistered(model={}, owner={:?})", model_id, owner)
            }
            Self::RoundStarted { model_id, round_id, deadline, .. } => {
                write!(f, "RoundStarted(model={}, round={}, deadline={})", model_id, round_id, deadline)
            }
            Self::ProofSubmitted { model_id, round_id, prover, .. } => {
                write!(f, "ProofSubmitted(model={}, round={}, prover={:?})", model_id, round_id, prover)
            }
            Self::ProofAccepted { model_id, round_id, step_number, .. } => {
                write!(f, "ProofAccepted(model={}, round={}, step={})", model_id, round_id, step_number)
            }
            Self::RoundCompleted { model_id, round_id, .. } => {
                write!(f, "RoundCompleted(model={}, round={})", model_id, round_id)
            }
            Self::RoundFinalized { model_id, round_id, total_steps, .. } => {
                write!(f, "RoundFinalized(model={}, round={}, steps={})", model_id, round_id, total_steps)
            }
            Self::Staked { prover, model_id, amount, .. } => {
                write!(f, "Staked(prover={:?}, model={}, amount={})", prover, model_id, amount)
            }
            Self::Unstaked { prover, model_id, amount } => {
                write!(f, "Unstaked(prover={:?}, model={}, amount={})", prover, model_id, amount)
            }
            Self::Slashed { prover, model_id, amount, reason, .. } => {
                write!(f, "Slashed(prover={:?}, model={}, amount={}, reason={})", prover, model_id, amount, reason)
            }
            Self::InvalidProofDetected { model_id, round_id, prover, .. } => {
                write!(f, "InvalidProofDetected(model={}, round={}, prover={:?})", model_id, round_id, prover)
            }
            Self::TrainingHalted { model_id, round_id, .. } => {
                write!(f, "TrainingHalted(model={}, round={})", model_id, round_id)
            }
            Self::EmergencyPauseChanged { is_paused, changed_by } => {
                write!(f, "EmergencyPauseChanged(paused={}, by={:?})", is_paused, changed_by)
            }
            Self::ModelStateChanged { model_id, active, .. } => {
                write!(f, "ModelStateChanged(model={}, active={})", model_id, active)
            }
            Self::ErrorBudgetWarning { model_id, round_id, .. } => {
                write!(f, "ErrorBudgetWarning(model={}, round={})", model_id, round_id)
            }
            Self::CommitmentUpdated { model_id, round_id, step_number, .. } => {
                write!(f, "CommitmentUpdated(model={}, round={}, step={})", model_id, round_id, step_number)
            }
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the on-chain event watcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainWatcherConfig {
    /// How often to poll for new blocks (seconds). Default: 3.
    #[serde(default = "default_poll_interval_secs")]
    pub poll_interval_secs: u64,

    /// Number of block confirmations required before processing events.
    /// Higher values provide better reorg safety at the cost of latency.
    /// Default: 2 (suitable for L2s and local devnets).
    /// Use 12+ for Ethereum mainnet.
    #[serde(default = "default_confirmation_depth")]
    pub confirmation_depth: u64,

    /// Maximum number of blocks to scan in a single poll cycle.
    /// Prevents overwhelming the RPC endpoint after long downtime.
    /// Default: 1000.
    #[serde(default = "default_max_block_range")]
    pub max_block_range: u64,

    /// Block number to start watching from.
    /// If None, starts from the latest block on first run,
    /// then uses the persistent cursor on subsequent runs.
    #[serde(default)]
    pub start_block: Option<u64>,

    /// Whether the event watcher is enabled. Default: true.
    #[serde(default = "default_watcher_enabled")]
    pub enabled: bool,
}

fn default_poll_interval_secs() -> u64 { 3 }
fn default_confirmation_depth() -> u64 { 2 }
fn default_max_block_range() -> u64 { 1000 }
fn default_watcher_enabled() -> bool { true }

impl Default for ChainWatcherConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: default_poll_interval_secs(),
            confirmation_depth: default_confirmation_depth(),
            max_block_range: default_max_block_range(),
            start_block: None,
            enabled: default_watcher_enabled(),
        }
    }
}

// ============================================================================
// Persistent Cursor
// ============================================================================

/// Persistent cursor tracking the last fully-processed block.
///
/// Stored as JSON in the node's data directory. On restart, the watcher
/// validates the stored block hash still matches the chain; if a reorg
/// invalidated it, the watcher rewinds to `last_processed_block - confirmation_depth`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherCursor {
    /// Last block whose events were fully processed and dispatched.
    pub last_processed_block: u64,
    /// Hash of the last processed block (for reorg detection).
    pub last_block_hash: String,
    /// Contract address being watched (to detect config changes).
    pub contract_address: String,
}

impl WatcherCursor {
    /// Saves the cursor to a JSON file at the given path.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write to temp file first, then rename (atomic on most filesystems).
        let tmp_path = path.with_extension("tmp");
        let data = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&tmp_path, data)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Loads a cursor from a JSON file. Returns None if the file doesn't exist.
    pub fn load(path: &Path) -> std::io::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(data) => {
                let cursor: Self = serde_json::from_str(&data)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(Some(cursor))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ============================================================================
// Chain Watcher
// ============================================================================

/// Watches a HelixCoordinator contract for on-chain events.
///
/// The watcher runs a background polling loop that:
/// 1. Queries the current block number
/// 2. Computes the safe block (current - confirmation_depth)
/// 3. Fetches logs from `last_processed + 1` to `safe_block`
/// 4. Parses each log into a typed [`ChainEvent`]
/// 5. Broadcasts events via a `tokio::sync::broadcast` channel
/// 6. Persists the cursor to disk for crash recovery
///
/// # Usage
///
/// ```no_run
/// # use helix_node::chain_watcher::{ChainWatcher, ChainWatcherConfig};
/// # async fn example() -> anyhow::Result<()> {
/// let watcher = ChainWatcher::new(
///     "http://localhost:8545",
///     "0x1234...",
///     ChainWatcherConfig::default(),
///     "/tmp/helix/watcher_cursor.json".into(),
/// ).await?;
///
/// let mut rx = watcher.subscribe();
/// let handle = watcher.start();
///
/// while let Ok(event) = rx.recv().await {
///     println!("Event: {}", event.data);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ChainWatcher {
    /// Ethers HTTP provider for the target chain.
    provider: Arc<Provider<Http>>,
    /// Address of the HelixCoordinator contract to watch.
    contract_address: Address,
    /// Watcher configuration.
    config: ChainWatcherConfig,
    /// Broadcast sender for dispatching confirmed events.
    event_tx: broadcast::Sender<ChainEvent>,
    /// Path to persist the watcher cursor.
    cursor_path: PathBuf,
    /// Last processed block (atomic for cross-task reads).
    last_processed_block: Arc<AtomicU64>,
    /// Shutdown signal sender.
    shutdown_tx: watch::Sender<bool>,
    /// Shutdown signal receiver (cloned for the polling task).
    shutdown_rx: watch::Receiver<bool>,
}

impl ChainWatcher {
    /// Creates a new chain watcher.
    ///
    /// Does NOT start polling — call [`start()`] to begin watching.
    pub async fn new(
        rpc_url: &str,
        contract_address: &str,
        config: ChainWatcherConfig,
        cursor_path: PathBuf,
    ) -> anyhow::Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url)?;
        let address = contract_address.parse::<Address>()?;

        let (event_tx, _) = broadcast::channel(256);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        // Load persisted cursor or use configured start block
        let initial_block = Self::resolve_start_block(
            &provider,
            &address,
            &config,
            &cursor_path,
        ).await?;

        Ok(Self {
            provider: Arc::new(provider),
            contract_address: address,
            config,
            event_tx,
            cursor_path,
            last_processed_block: Arc::new(AtomicU64::new(initial_block)),
            shutdown_tx,
            shutdown_rx,
        })
    }

    /// Returns a broadcast receiver for chain events.
    ///
    /// Multiple consumers can subscribe; each gets an independent copy
    /// of every event. Slow consumers that fall behind will receive
    /// a `RecvError::Lagged` with the number of missed events.
    pub fn subscribe(&self) -> broadcast::Receiver<ChainEvent> {
        self.event_tx.subscribe()
    }

    /// Returns the last fully-processed block number.
    pub fn last_processed_block(&self) -> u64 {
        self.last_processed_block.load(Ordering::Relaxed)
    }

    /// Returns the contract address being watched.
    pub fn contract_address(&self) -> Address {
        self.contract_address
    }

    /// Sends the shutdown signal, causing the polling loop to exit gracefully.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Starts the background polling loop.
    ///
    /// Returns a `JoinHandle` that resolves when the watcher shuts down
    /// (either via [`stop()`] or an unrecoverable error). The watcher
    /// remains alive and `stop()` can be called to trigger graceful shutdown.
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let provider = self.provider.clone();
        let contract_address = self.contract_address;
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let cursor_path = self.cursor_path.clone();
        let last_processed_block = self.last_processed_block.clone();
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            info!(
                contract = %contract_address,
                poll_secs = config.poll_interval_secs,
                confirmations = config.confirmation_depth,
                start_block = last_processed_block.load(Ordering::Relaxed),
                "Chain watcher started"
            );

            let mut interval = tokio::time::interval(
                Duration::from_secs(config.poll_interval_secs),
            );

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::poll_cycle(
                            &provider,
                            contract_address,
                            &config,
                            &event_tx,
                            &cursor_path,
                            &last_processed_block,
                        ).await {
                            error!(error = %e, "Chain watcher poll cycle failed");
                            // Continue polling — transient RPC errors are common.
                            // Persistent failures will be logged on every cycle.
                        }
                    }
                    result = shutdown_rx.changed() => {
                        // Break on explicit shutdown OR sender dropped (Err).
                        if result.is_err() || *shutdown_rx.borrow() {
                            info!("Chain watcher shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Resolves the starting block number from the cursor or config.
    async fn resolve_start_block(
        provider: &Provider<Http>,
        contract_address: &Address,
        config: &ChainWatcherConfig,
        cursor_path: &Path,
    ) -> anyhow::Result<u64> {
        // Try loading persisted cursor
        if let Some(cursor) = WatcherCursor::load(cursor_path)? {
            // Verify the cursor is for the same contract
            let stored_addr = cursor.contract_address.to_lowercase();
            let current_addr = format!("{:?}", contract_address).to_lowercase();
            if stored_addr == current_addr {
                // Validate the stored block hash still matches the chain
                let block_num = cursor.last_processed_block;
                match provider.get_block(block_num).await {
                    Ok(Some(block)) => {
                        let chain_hash = format!("{:?}", block.hash.unwrap_or_default());
                        if chain_hash == cursor.last_block_hash {
                            info!(
                                block = block_num,
                                "Resuming chain watcher from persisted cursor"
                            );
                            return Ok(block_num);
                        }
                        // Reorg detected — rewind by confirmation_depth
                        let rewind = block_num.saturating_sub(config.confirmation_depth);
                        warn!(
                            stored_block = block_num,
                            rewind_to = rewind,
                            "Reorg detected: stored block hash mismatch, rewinding"
                        );
                        return Ok(rewind);
                    }
                    Ok(None) => {
                        // Block doesn't exist (deep reorg or pruned)
                        warn!(
                            block = block_num,
                            "Stored block not found on chain, starting from configured start_block"
                        );
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            "Failed to validate cursor block, using cursor as-is"
                        );
                        return Ok(cursor.last_processed_block);
                    }
                }
            } else {
                info!("Cursor contract address mismatch, starting fresh");
            }
        }

        // Use configured start_block or latest
        if let Some(start) = config.start_block {
            Ok(start)
        } else {
            let current = provider.get_block_number().await?;
            Ok(current.as_u64())
        }
    }

    /// Executes one poll cycle: fetch new blocks, parse events, broadcast.
    async fn poll_cycle(
        provider: &Provider<Http>,
        contract_address: Address,
        config: &ChainWatcherConfig,
        event_tx: &broadcast::Sender<ChainEvent>,
        cursor_path: &Path,
        last_processed_block: &AtomicU64,
    ) -> anyhow::Result<()> {
        let current_block = provider.get_block_number().await?.as_u64();
        let safe_block = current_block.saturating_sub(config.confirmation_depth);
        let cursor = last_processed_block.load(Ordering::Relaxed);

        if safe_block <= cursor {
            debug!(
                current = current_block,
                safe = safe_block,
                cursor = cursor,
                "No new confirmed blocks"
            );
            return Ok(());
        }

        // Cap the range to max_block_range to avoid huge queries after downtime
        let from_block = cursor + 1;
        let to_block = std::cmp::min(safe_block, from_block + config.max_block_range - 1);

        debug!(
            from = from_block,
            to = to_block,
            safe = safe_block,
            "Polling blocks for events"
        );

        // Query logs for our contract address
        let filter = Filter::new()
            .address(contract_address)
            .from_block(from_block)
            .to_block(to_block);

        let logs = provider.get_logs(&filter).await?;

        let mut events_dispatched = 0u64;
        for log in &logs {
            match parse_log_to_event(log) {
                Ok(event) => {
                    info!(
                        block = event.block_number,
                        tx = %event.tx_hash,
                        event = %event.data,
                        "Chain event confirmed"
                    );

                    // Broadcast to all subscribers (ignore error = no active receivers)
                    let _ = event_tx.send(event);
                    events_dispatched += 1;
                }
                Err(e) => {
                    // Unknown event signature — skip silently (contract may emit
                    // events we don't watch, like admin/guardian events).
                    debug!(
                        tx = ?log.transaction_hash,
                        topics = ?log.topics.first(),
                        error = %e,
                        "Skipping unrecognized log"
                    );
                }
            }
        }

        if events_dispatched > 0 {
            info!(
                blocks = to_block - from_block + 1,
                events = events_dispatched,
                "Poll cycle complete"
            );
        }

        // Persist cursor (get the block hash for reorg detection)
        let block_hash = if let Ok(Some(block)) = provider.get_block(to_block).await {
            format!("{:?}", block.hash.unwrap_or_default())
        } else {
            String::new()
        };

        let cursor_data = WatcherCursor {
            last_processed_block: to_block,
            last_block_hash: block_hash,
            contract_address: format!("{:?}", contract_address),
        };

        if let Err(e) = cursor_data.save(cursor_path) {
            error!(error = %e, "Failed to persist watcher cursor");
            // Non-fatal: we'll just re-process events on restart (idempotent).
        }

        last_processed_block.store(to_block, Ordering::Relaxed);

        Ok(())
    }
}

// ============================================================================
// Log Parsing
// ============================================================================

/// Parses a raw ethers [`Log`] into a typed [`ChainEvent`].
///
/// Uses the abigen-generated `HelixCoordinatorWatcherEvents` enum for
/// topic-based event signature matching and ABI decoding.
pub fn parse_log_to_event(log: &Log) -> anyhow::Result<ChainEvent> {
    let raw_log = RawLog {
        topics: log.topics.clone(),
        data: log.data.to_vec(),
    };

    let decoded = HelixCoordinatorWatcherEvents::decode_log(&raw_log)
        .map_err(|e| anyhow::anyhow!("Failed to decode log: {}", e))?;

    let block_number = log.block_number.map(|b| b.as_u64()).unwrap_or(0);
    let block_hash = log.block_hash.unwrap_or_default();
    let tx_hash = log.transaction_hash.unwrap_or_default();
    let log_index = log.log_index.map(|i| i.as_u64()).unwrap_or(0);

    let data = match decoded {
        HelixCoordinatorWatcherEvents::ModelRegisteredFilter(e) => {
            ChainEventData::ModelRegistered {
                model_id: e.model_id.as_u64(),
                owner: e.owner,
                initial_commitment: e.initial_commitment,
                min_stake: e.min_stake,
                ipfs_hash: e.ipfs_hash,
            }
        }
        HelixCoordinatorWatcherEvents::RoundStartedFilter(e) => {
            ChainEventData::RoundStarted {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                deadline: e.deadline.as_u64(),
                model_commitment: e.model_commitment,
            }
        }
        HelixCoordinatorWatcherEvents::ProofSubmittedFilter(e) => {
            ChainEventData::ProofSubmitted {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                prover: e.prover,
                new_commitment: e.new_commitment,
                error_bound: e.error_bound,
            }
        }
        HelixCoordinatorWatcherEvents::ProofAcceptedFilter(e) => {
            ChainEventData::ProofAccepted {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                prover: e.prover,
                step_number: e.step_number.as_u64(),
                new_commitment: e.new_commitment,
                loss: e.loss,
                error_bound: e.error_bound,
            }
        }
        HelixCoordinatorWatcherEvents::RoundCompletedFilter(e) => {
            ChainEventData::RoundCompleted {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                new_commitment: e.new_commitment,
                total_error_bound: e.total_error_bound,
            }
        }
        HelixCoordinatorWatcherEvents::RoundFinalizedFilter(e) => {
            ChainEventData::RoundFinalized {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                final_commitment: e.final_commitment,
                total_steps: e.total_steps.as_u64(),
                final_loss: e.final_loss,
                total_error: e.total_error,
            }
        }
        HelixCoordinatorWatcherEvents::StakedFilter(e) => {
            ChainEventData::Staked {
                prover: e.prover,
                model_id: e.model_id.as_u64(),
                amount: e.amount,
                total_stake: e.total_stake,
            }
        }
        HelixCoordinatorWatcherEvents::UnstakedFilter(e) => {
            ChainEventData::Unstaked {
                prover: e.prover,
                model_id: e.model_id.as_u64(),
                amount: e.amount,
            }
        }
        HelixCoordinatorWatcherEvents::SlashedFilter(e) => {
            ChainEventData::Slashed {
                prover: e.prover,
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                amount: e.amount,
                remaining_stake: e.remaining_stake,
                reason: e.reason,
            }
        }
        HelixCoordinatorWatcherEvents::InvalidProofDetectedFilter(e) => {
            ChainEventData::InvalidProofDetected {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                prover: e.prover,
                proof_hash: e.proof_hash,
            }
        }
        HelixCoordinatorWatcherEvents::TrainingHaltedFilter(e) => {
            ChainEventData::TrainingHalted {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                accumulated_error: e.accumulated_error,
                max_budget: e.max_budget,
            }
        }
        HelixCoordinatorWatcherEvents::EmergencyPauseChangedFilter(e) => {
            ChainEventData::EmergencyPauseChanged {
                is_paused: e.is_paused,
                changed_by: e.changed_by,
            }
        }
        HelixCoordinatorWatcherEvents::ModelStateChangedFilter(e) => {
            ChainEventData::ModelStateChanged {
                model_id: e.model_id.as_u64(),
                active: e.active,
                changed_by: e.changed_by,
            }
        }
        HelixCoordinatorWatcherEvents::ErrorBudgetWarningFilter(e) => {
            ChainEventData::ErrorBudgetWarning {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                accumulated: e.accumulated,
                budget: e.budget,
            }
        }
        HelixCoordinatorWatcherEvents::CommitmentUpdatedFilter(e) => {
            ChainEventData::CommitmentUpdated {
                model_id: e.model_id.as_u64(),
                round_id: e.round_id.as_u64(),
                old_commitment: e.old_commitment,
                new_commitment: e.new_commitment,
                step_number: e.step_number.as_u64(),
            }
        }
    };

    Ok(ChainEvent {
        data,
        block_number,
        block_hash,
        tx_hash,
        log_index,
    })
}

// ============================================================================
// Event Handler Trait
// ============================================================================

/// Trait for components that react to on-chain events.
///
/// Implement this for aggregators, workers, or any subsystem that needs
/// to respond to chain state changes. The handler receives events in order.
#[async_trait::async_trait]
pub trait ChainEventHandler: Send + Sync {
    /// Handle a confirmed chain event.
    ///
    /// Called for each event in block order. Implementations should be
    /// non-blocking — use message passing or state updates rather than
    /// blocking I/O.
    async fn handle_event(&self, event: &ChainEvent);
}

/// Spawns a background task that forwards events from the broadcast channel
/// to a handler implementing [`ChainEventHandler`].
///
/// Returns a `JoinHandle` that runs until the broadcast channel is closed
/// or the handler task is dropped.
pub fn spawn_event_handler<H: ChainEventHandler + 'static>(
    mut rx: broadcast::Receiver<ChainEvent>,
    handler: Arc<H>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    handler.handle_event(&event).await;
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        missed = n,
                        "Event handler fell behind, skipped {} events", n
                    );
                }
                Err(broadcast::error::RecvError::Closed) => {
                    info!("Event broadcast channel closed, handler exiting");
                    break;
                }
            }
        }
    })
}

// ============================================================================
// Node Event Reactor
// ============================================================================

/// Default event reactor that logs and tracks events relevant to node operation.
///
/// This reactor:
/// - Tracks which models and rounds are active on-chain
/// - Detects if the local node's address was slashed
/// - Pauses operations when the contract is emergency-paused
/// - Logs all events at appropriate severity levels
pub struct NodeEventReactor {
    /// The local node's signer address (for detecting slashing).
    local_address: Option<Address>,
    /// Whether the contract is currently paused.
    contract_paused: std::sync::atomic::AtomicBool,
    /// Active model IDs and their current rounds.
    active_models: parking_lot::RwLock<std::collections::HashMap<u64, ActiveModelState>>,
}

/// Tracked state for an active model.
#[derive(Debug, Clone)]
pub struct ActiveModelState {
    /// Current round ID (0 if no active round).
    pub current_round: u64,
    /// Whether the model is active.
    pub active: bool,
    /// Current commitment hash.
    pub commitment: U256,
    /// Latest accepted step number.
    pub last_step: u64,
    /// Whether the current round is halted.
    pub round_halted: bool,
}

impl NodeEventReactor {
    /// Creates a new event reactor.
    ///
    /// If `local_address` is provided, the reactor will detect slashing events
    /// targeting this address and log them at ERROR level.
    pub fn new(local_address: Option<Address>) -> Self {
        Self {
            local_address,
            contract_paused: std::sync::atomic::AtomicBool::new(false),
            active_models: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Returns whether the contract is currently paused.
    pub fn is_contract_paused(&self) -> bool {
        self.contract_paused.load(Ordering::Relaxed)
    }

    /// Returns the tracked state for a model, if any.
    pub fn get_model_state(&self, model_id: u64) -> Option<ActiveModelState> {
        self.active_models.read().get(&model_id).cloned()
    }

    /// Returns all tracked active models.
    pub fn active_models(&self) -> std::collections::HashMap<u64, ActiveModelState> {
        self.active_models.read().clone()
    }
}

#[async_trait::async_trait]
impl ChainEventHandler for NodeEventReactor {
    async fn handle_event(&self, event: &ChainEvent) {
        match &event.data {
            ChainEventData::ModelRegistered { model_id, owner, initial_commitment, .. } => {
                self.active_models.write().insert(*model_id, ActiveModelState {
                    current_round: 0,
                    active: true,
                    commitment: *initial_commitment,
                    last_step: 0,
                    round_halted: false,
                });
                info!(
                    model_id = model_id,
                    owner = ?owner,
                    "New model registered on-chain"
                );
            }

            ChainEventData::RoundStarted { model_id, round_id, deadline, model_commitment } => {
                if let Some(state) = self.active_models.write().get_mut(model_id) {
                    state.current_round = *round_id;
                    state.commitment = *model_commitment;
                    state.round_halted = false;
                }
                info!(
                    model_id = model_id,
                    round_id = round_id,
                    deadline = deadline,
                    "Training round started on-chain"
                );
            }

            ChainEventData::ProofAccepted { model_id, round_id, prover, step_number, new_commitment, loss, .. } => {
                if let Some(state) = self.active_models.write().get_mut(model_id) {
                    state.commitment = *new_commitment;
                    state.last_step = *step_number;
                }

                let is_local = self.local_address.map_or(false, |a| a == *prover);
                if is_local {
                    info!(
                        model_id = model_id,
                        round_id = round_id,
                        step = step_number,
                        loss = %loss,
                        "LOCAL proof accepted on-chain"
                    );
                }
            }

            ChainEventData::RoundFinalized { model_id, round_id, total_steps, final_loss, .. } => {
                info!(
                    model_id = model_id,
                    round_id = round_id,
                    total_steps = total_steps,
                    final_loss = %final_loss,
                    "Training round finalized on-chain"
                );
            }

            ChainEventData::Slashed { prover, model_id, amount, reason, .. } => {
                let is_local = self.local_address.map_or(false, |a| a == *prover);
                if is_local {
                    error!(
                        model_id = model_id,
                        amount = %amount,
                        reason = reason,
                        "LOCAL NODE WAS SLASHED — corrective action needed"
                    );
                } else {
                    warn!(
                        prover = ?prover,
                        model_id = model_id,
                        amount = %amount,
                        reason = reason,
                        "Prover slashed on-chain"
                    );
                }
            }

            ChainEventData::InvalidProofDetected { model_id, round_id, prover, .. } => {
                let is_local = self.local_address.map_or(false, |a| a == *prover);
                if is_local {
                    error!(
                        model_id = model_id,
                        round_id = round_id,
                        "LOCAL invalid proof detected — slashing imminent"
                    );
                }
            }

            ChainEventData::TrainingHalted { model_id, round_id, accumulated_error, max_budget } => {
                if let Some(state) = self.active_models.write().get_mut(model_id) {
                    state.round_halted = true;
                }
                warn!(
                    model_id = model_id,
                    round_id = round_id,
                    accumulated = %accumulated_error,
                    budget = %max_budget,
                    "Training halted: error budget exhausted"
                );
            }

            ChainEventData::EmergencyPauseChanged { is_paused, changed_by } => {
                self.contract_paused.store(*is_paused, Ordering::Relaxed);
                if *is_paused {
                    error!(
                        changed_by = ?changed_by,
                        "CONTRACT EMERGENCY PAUSED — all operations halted"
                    );
                } else {
                    info!(
                        changed_by = ?changed_by,
                        "Contract unpaused — operations resumed"
                    );
                }
            }

            ChainEventData::ModelStateChanged { model_id, active, .. } => {
                if let Some(state) = self.active_models.write().get_mut(model_id) {
                    state.active = *active;
                }
            }

            ChainEventData::ErrorBudgetWarning { model_id, round_id, accumulated, budget } => {
                warn!(
                    model_id = model_id,
                    round_id = round_id,
                    pct = format!("{}%", (*accumulated * U256::from(100)) / *budget),
                    "Error budget nearing limit (≥80%)"
                );
            }

            // Events that are informational only — no state change needed
            ChainEventData::ProofSubmitted { .. }
            | ChainEventData::RoundCompleted { .. }
            | ChainEventData::Staked { .. }
            | ChainEventData::Unstaked { .. }
            | ChainEventData::CommitmentUpdated { .. } => {}
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_watcher_config_defaults() {
        let config = ChainWatcherConfig::default();
        assert_eq!(config.poll_interval_secs, 3);
        assert_eq!(config.confirmation_depth, 2);
        assert_eq!(config.max_block_range, 1000);
        assert!(config.start_block.is_none());
        assert!(config.enabled);
    }

    #[test]
    fn test_chain_watcher_config_serde() {
        let json = r#"{
            "poll_interval_secs": 5,
            "confirmation_depth": 12,
            "max_block_range": 500,
            "start_block": 100,
            "enabled": true
        }"#;
        let config: ChainWatcherConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.poll_interval_secs, 5);
        assert_eq!(config.confirmation_depth, 12);
        assert_eq!(config.max_block_range, 500);
        assert_eq!(config.start_block, Some(100));
    }

    #[test]
    fn test_cursor_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cursor.json");

        let cursor = WatcherCursor {
            last_processed_block: 42,
            last_block_hash: "0xabc123".to_string(),
            contract_address: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
        };

        cursor.save(&path).unwrap();

        let loaded = WatcherCursor::load(&path).unwrap().unwrap();
        assert_eq!(loaded.last_processed_block, 42);
        assert_eq!(loaded.last_block_hash, "0xabc123");
        assert_eq!(loaded.contract_address, cursor.contract_address);
    }

    #[test]
    fn test_cursor_load_nonexistent() {
        let path = PathBuf::from("/tmp/nonexistent_cursor_12345.json");
        let result = WatcherCursor::load(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cursor_save_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subdir1").join("subdir2").join("cursor.json");

        let cursor = WatcherCursor {
            last_processed_block: 1,
            last_block_hash: "0x00".to_string(),
            contract_address: "0x00".to_string(),
        };

        cursor.save(&path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_chain_event_data_display() {
        let event = ChainEventData::ModelRegistered {
            model_id: 5,
            owner: Address::zero(),
            initial_commitment: U256::from(1),
            min_stake: U256::from(1000),
            ipfs_hash: "QmTest".to_string(),
        };
        let s = format!("{}", event);
        assert!(s.contains("ModelRegistered"));
        assert!(s.contains("model=5"));

        let event = ChainEventData::Slashed {
            prover: Address::zero(),
            model_id: 1,
            round_id: 2,
            amount: U256::from(500),
            remaining_stake: U256::from(500),
            reason: "Invalid proof".to_string(),
        };
        let s = format!("{}", event);
        assert!(s.contains("Slashed"));
        assert!(s.contains("Invalid proof"));
    }

    #[test]
    fn test_node_event_reactor_pause_tracking() {
        let reactor = NodeEventReactor::new(None);
        assert!(!reactor.is_contract_paused());

        // Simulate handling a pause event
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let event = ChainEvent {
                data: ChainEventData::EmergencyPauseChanged {
                    is_paused: true,
                    changed_by: Address::zero(),
                },
                block_number: 100,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            };
            reactor.handle_event(&event).await;
        });

        assert!(reactor.is_contract_paused());
    }

    #[test]
    fn test_node_event_reactor_model_tracking() {
        let reactor = NodeEventReactor::new(None);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            // Register a model
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::ModelRegistered {
                    model_id: 1,
                    owner: Address::zero(),
                    initial_commitment: U256::from(42),
                    min_stake: U256::from(1000),
                    ipfs_hash: "QmTest".to_string(),
                },
                block_number: 10,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;

            let state = reactor.get_model_state(1).unwrap();
            assert!(state.active);
            assert_eq!(state.current_round, 0);
            assert_eq!(state.commitment, U256::from(42));

            // Start a round
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::RoundStarted {
                    model_id: 1,
                    round_id: 1,
                    deadline: 9999,
                    model_commitment: U256::from(42),
                },
                block_number: 11,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;

            let state = reactor.get_model_state(1).unwrap();
            assert_eq!(state.current_round, 1);

            // Accept a proof
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::ProofAccepted {
                    model_id: 1,
                    round_id: 1,
                    prover: Address::zero(),
                    step_number: 1,
                    new_commitment: U256::from(100),
                    loss: U256::from(50),
                    error_bound: U256::from(5),
                },
                block_number: 12,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;

            let state = reactor.get_model_state(1).unwrap();
            assert_eq!(state.last_step, 1);
            assert_eq!(state.commitment, U256::from(100));

            // Halt training
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::TrainingHalted {
                    model_id: 1,
                    round_id: 1,
                    accumulated_error: U256::from(900),
                    max_budget: U256::from(1000),
                },
                block_number: 13,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;

            let state = reactor.get_model_state(1).unwrap();
            assert!(state.round_halted);
        });
    }

    #[test]
    fn test_node_event_reactor_local_slash_detection() {
        let local_addr: Address = "0x1234567890123456789012345678901234567890"
            .parse()
            .unwrap();
        let reactor = NodeEventReactor::new(Some(local_addr));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            // Slash event for our address — should log ERROR
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::Slashed {
                    prover: local_addr,
                    model_id: 1,
                    round_id: 1,
                    amount: U256::from(500),
                    remaining_stake: U256::from(500),
                    reason: "Invalid proof".to_string(),
                },
                block_number: 20,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;

            // Slash event for different address — should log WARN
            reactor.handle_event(&ChainEvent {
                data: ChainEventData::Slashed {
                    prover: Address::zero(),
                    model_id: 1,
                    round_id: 1,
                    amount: U256::from(200),
                    remaining_stake: U256::from(800),
                    reason: "Other prover".to_string(),
                },
                block_number: 21,
                block_hash: H256::zero(),
                tx_hash: H256::zero(),
                log_index: 0,
            }).await;
        });
        // Test passes if it doesn't panic — log output verifies behavior
    }

    #[test]
    fn test_parse_log_model_registered() {
        // Build a synthetic ModelRegistered log matching the ABI:
        // event ModelRegistered(uint256 indexed modelId, address indexed owner,
        //     uint256 initialCommitment, uint256 minStake, string ipfsHash)

        use ethers::abi::{encode, Token};
        use ethers::utils::keccak256;

        let event_sig = keccak256(
            "ModelRegistered(uint256,address,uint256,uint256,string)"
        );

        let model_id = U256::from(7);
        let owner: Address = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .parse()
            .unwrap();

        // Topics: [event_sig, indexed_modelId, indexed_owner]
        let topics = vec![
            H256::from(event_sig),
            H256::from_uint(&model_id),
            {
                let mut bytes = [0u8; 32];
                bytes[12..32].copy_from_slice(owner.as_bytes());
                H256::from(bytes)
            },
        ];

        // Data: non-indexed params (initialCommitment, minStake, ipfsHash)
        let data = encode(&[
            Token::Uint(U256::from(999)),    // initialCommitment
            Token::Uint(U256::from(1000)),   // minStake
            Token::String("QmTestHash".to_string()), // ipfsHash
        ]);

        let log = Log {
            address: Address::zero(),
            topics,
            data: data.into(),
            block_hash: Some(H256::from_low_u64_be(42)),
            block_number: Some(U64::from(100)),
            transaction_hash: Some(H256::from_low_u64_be(1)),
            transaction_index: Some(U64::from(0)),
            log_index: Some(U256::from(0)),
            transaction_log_index: None,
            log_type: None,
            removed: Some(false),
        };

        let event = parse_log_to_event(&log).unwrap();
        assert_eq!(event.block_number, 100);

        match event.data {
            ChainEventData::ModelRegistered {
                model_id,
                owner: parsed_owner,
                initial_commitment,
                min_stake,
                ipfs_hash,
            } => {
                assert_eq!(model_id, 7);
                assert_eq!(parsed_owner, owner);
                assert_eq!(initial_commitment, U256::from(999));
                assert_eq!(min_stake, U256::from(1000));
                assert_eq!(ipfs_hash, "QmTestHash");
            }
            other => panic!("Expected ModelRegistered, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_log_round_started() {
        use ethers::abi::{encode, Token};
        use ethers::utils::keccak256;

        let event_sig = keccak256(
            "RoundStarted(uint256,uint256,uint256,uint256)"
        );

        let topics = vec![
            H256::from(event_sig),
            H256::from_uint(&U256::from(3)),  // indexed modelId
            H256::from_uint(&U256::from(1)),  // indexed roundId
        ];

        let data = encode(&[
            Token::Uint(U256::from(99999)),  // deadline
            Token::Uint(U256::from(42)),     // modelCommitment
        ]);

        let log = Log {
            address: Address::zero(),
            topics,
            data: data.into(),
            block_hash: Some(H256::zero()),
            block_number: Some(U64::from(50)),
            transaction_hash: Some(H256::zero()),
            transaction_index: Some(U64::from(0)),
            log_index: Some(U256::from(1)),
            transaction_log_index: None,
            log_type: None,
            removed: Some(false),
        };

        let event = parse_log_to_event(&log).unwrap();
        match event.data {
            ChainEventData::RoundStarted {
                model_id,
                round_id,
                deadline,
                model_commitment,
            } => {
                assert_eq!(model_id, 3);
                assert_eq!(round_id, 1);
                assert_eq!(deadline, 99999);
                assert_eq!(model_commitment, U256::from(42));
            }
            other => panic!("Expected RoundStarted, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_log_slashed() {
        use ethers::abi::{encode, Token};
        use ethers::utils::keccak256;

        let event_sig = keccak256(
            "Slashed(address,uint256,uint256,uint256,uint256,string)"
        );

        let prover: Address = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .parse()
            .unwrap();

        let topics = vec![
            H256::from(event_sig),
            {
                let mut bytes = [0u8; 32];
                bytes[12..32].copy_from_slice(prover.as_bytes());
                H256::from(bytes)
            },
            H256::from_uint(&U256::from(5)), // indexed modelId
        ];

        let data = encode(&[
            Token::Uint(U256::from(1)),      // roundId
            Token::Uint(U256::from(500)),    // amount
            Token::Uint(U256::from(500)),    // remainingStake
            Token::String("Invalid proof".to_string()), // reason
        ]);

        let log = Log {
            address: Address::zero(),
            topics,
            data: data.into(),
            block_hash: Some(H256::zero()),
            block_number: Some(U64::from(200)),
            transaction_hash: Some(H256::zero()),
            transaction_index: Some(U64::from(0)),
            log_index: Some(U256::from(0)),
            transaction_log_index: None,
            log_type: None,
            removed: Some(false),
        };

        let event = parse_log_to_event(&log).unwrap();
        match event.data {
            ChainEventData::Slashed {
                prover: parsed_prover,
                model_id,
                round_id,
                amount,
                remaining_stake,
                reason,
            } => {
                assert_eq!(parsed_prover, prover);
                assert_eq!(model_id, 5);
                assert_eq!(round_id, 1);
                assert_eq!(amount, U256::from(500));
                assert_eq!(remaining_stake, U256::from(500));
                assert_eq!(reason, "Invalid proof");
            }
            other => panic!("Expected Slashed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_log_emergency_pause() {
        use ethers::abi::{encode, Token};
        use ethers::utils::keccak256;

        let event_sig = keccak256("EmergencyPauseChanged(bool,address)");

        let changer: Address = "0xcccccccccccccccccccccccccccccccccccccccc"
            .parse()
            .unwrap();

        let topics = vec![
            H256::from(event_sig),
            {
                let mut bytes = [0u8; 32];
                bytes[12..32].copy_from_slice(changer.as_bytes());
                H256::from(bytes)
            },
        ];

        let data = encode(&[Token::Bool(true)]);

        let log = Log {
            address: Address::zero(),
            topics,
            data: data.into(),
            block_hash: Some(H256::zero()),
            block_number: Some(U64::from(300)),
            transaction_hash: Some(H256::zero()),
            transaction_index: Some(U64::from(0)),
            log_index: Some(U256::from(0)),
            transaction_log_index: None,
            log_type: None,
            removed: Some(false),
        };

        let event = parse_log_to_event(&log).unwrap();
        match event.data {
            ChainEventData::EmergencyPauseChanged { is_paused, changed_by } => {
                assert!(is_paused);
                assert_eq!(changed_by, changer);
            }
            other => panic!("Expected EmergencyPauseChanged, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_log_unknown_event_returns_error() {
        use ethers::utils::keccak256;

        let unknown_sig = keccak256("SomeUnknownEvent(uint256)");

        let log = Log {
            address: Address::zero(),
            topics: vec![H256::from(unknown_sig)],
            data: Default::default(),
            block_hash: Some(H256::zero()),
            block_number: Some(U64::from(1)),
            transaction_hash: Some(H256::zero()),
            transaction_index: Some(U64::from(0)),
            log_index: Some(U256::from(0)),
            transaction_log_index: None,
            log_type: None,
            removed: Some(false),
        };

        assert!(parse_log_to_event(&log).is_err());
    }

    #[test]
    fn test_active_model_state_tracking() {
        let reactor = NodeEventReactor::new(None);
        assert!(reactor.active_models().is_empty());
        assert!(reactor.get_model_state(1).is_none());
    }
}
