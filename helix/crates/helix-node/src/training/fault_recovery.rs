//! Fault Recovery for Distributed Training.
//!
//! Provides concrete recovery mechanisms that build on top of the failure
//! detection infrastructure in `fault_tolerance.rs`:
//!
//! - **RetryPolicy**: Exponential backoff with jitter for retryable operations
//! - **ProofRetryManager**: Tracks and manages proof generation retries per worker
//! - **TransactionRetryManager**: Chain transaction submission with exponential backoff
//! - **WorkerFailureHandler**: Maps worker failures to round-level recovery actions
//! - **GracefulShutdownCoordinator**: SIGTERM/SIGINT handler that drains active rounds

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

use crate::network::messages::PeerId;
use super::job_manager::JobPhase;

// ============================================================================
// RetryPolicy
// ============================================================================

/// Configurable retry policy with exponential backoff and jitter.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Base delay between retries.
    pub base_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Backoff multiplier (typically 2.0).
    pub backoff_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_factor: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Creates a policy for proof generation retries.
    pub fn for_proof_generation() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
        }
    }

    /// Creates a policy for chain transaction retries.
    pub fn for_chain_transaction() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(120),
            backoff_factor: 2.0,
        }
    }

    /// Computes the delay for a given attempt number (0-indexed).
    ///
    /// Uses exponential backoff: `base_delay * backoff_factor^attempt`,
    /// capped at `max_delay`, with ±25% jitter to avoid thundering herd.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let delay_ms = base_ms * self.backoff_factor.powi(attempt as i32);
        let capped_ms = delay_ms.min(self.max_delay.as_millis() as f64);

        // Add ±25% jitter using a simple deterministic hash of the attempt
        // (we don't need cryptographic randomness for jitter)
        let jitter_factor = 0.75 + 0.5 * ((attempt as f64 * 0.618033988) % 1.0);
        let final_ms = (capped_ms * jitter_factor) as u64;

        Duration::from_millis(final_ms.max(1))
    }

    /// Returns whether the given attempt (0-indexed) should be retried.
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_retries
    }
}

// ============================================================================
// ProofRetryManager
// ============================================================================

/// Tracks proof generation retry state per worker per round.
#[derive(Debug, Clone)]
struct ProofAttemptState {
    round_id: u64,
    attempts: u32,
    last_error: Option<String>,
    last_attempt_at: Instant,
    succeeded: bool,
}

/// Manages proof generation retries across workers.
pub struct ProofRetryManager {
    policy: RetryPolicy,
    /// Per-worker retry state, keyed by (peer_id).
    /// Only tracks the current round per worker.
    attempts: RwLock<HashMap<PeerId, ProofAttemptState>>,
}

impl ProofRetryManager {
    /// Creates a new proof retry manager with the given policy.
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            attempts: RwLock::new(HashMap::new()),
        }
    }

    /// Creates a manager with default proof retry policy.
    pub fn default_policy() -> Self {
        Self::new(RetryPolicy::for_proof_generation())
    }

    /// Registers a proof generation attempt for a worker.
    pub fn register_attempt(&self, worker: &PeerId, round_id: u64) {
        let mut attempts = self.attempts.write();
        let state = attempts.entry(worker.clone()).or_insert_with(|| ProofAttemptState {
            round_id,
            attempts: 0,
            last_error: None,
            last_attempt_at: Instant::now(),
            succeeded: false,
        });

        // Reset if round changed
        if state.round_id != round_id {
            *state = ProofAttemptState {
                round_id,
                attempts: 0,
                last_error: None,
                last_attempt_at: Instant::now(),
                succeeded: false,
            };
        }

        state.attempts += 1;
        state.last_attempt_at = Instant::now();
        debug!(
            worker = %worker,
            round_id,
            attempt = state.attempts,
            "Proof generation attempt registered"
        );
    }

    /// Records a successful proof generation.
    pub fn record_success(&self, worker: &PeerId, round_id: u64) {
        let mut attempts = self.attempts.write();
        if let Some(state) = attempts.get_mut(worker) {
            if state.round_id == round_id {
                state.succeeded = true;
                info!(
                    worker = %worker,
                    round_id,
                    attempts = state.attempts,
                    "Proof generation succeeded"
                );
            }
        }
    }

    /// Records a failed proof generation attempt.
    pub fn record_failure(&self, worker: &PeerId, round_id: u64, error: String) {
        let mut attempts = self.attempts.write();
        if let Some(state) = attempts.get_mut(worker) {
            if state.round_id == round_id {
                state.last_error = Some(error.clone());
                warn!(
                    worker = %worker,
                    round_id,
                    attempt = state.attempts,
                    max = self.policy.max_retries,
                    error = %error,
                    "Proof generation failed"
                );
            }
        }
    }

    /// Returns whether a worker should retry proof generation for the given round.
    pub fn should_retry(&self, worker: &PeerId, round_id: u64) -> bool {
        let attempts = self.attempts.read();
        match attempts.get(worker) {
            Some(state) if state.round_id == round_id => {
                !state.succeeded && self.policy.should_retry(state.attempts)
            }
            _ => true, // No state = hasn't tried yet
        }
    }

    /// Returns the delay before the next retry attempt.
    pub fn next_retry_delay(&self, worker: &PeerId, round_id: u64) -> Duration {
        let attempts = self.attempts.read();
        match attempts.get(worker) {
            Some(state) if state.round_id == round_id => {
                self.policy.delay_for_attempt(state.attempts.saturating_sub(1))
            }
            _ => self.policy.base_delay,
        }
    }

    /// Returns workers that have permanently failed (exhausted all retries) for a round.
    pub fn permanently_failed(&self, round_id: u64) -> Vec<PeerId> {
        self.attempts
            .read()
            .iter()
            .filter(|(_, state)| {
                state.round_id == round_id
                    && !state.succeeded
                    && !self.policy.should_retry(state.attempts)
            })
            .map(|(peer_id, _)| peer_id.clone())
            .collect()
    }

    /// Returns the current attempt count for a worker in a round.
    pub fn attempt_count(&self, worker: &PeerId, round_id: u64) -> u32 {
        self.attempts
            .read()
            .get(worker)
            .filter(|s| s.round_id == round_id)
            .map(|s| s.attempts)
            .unwrap_or(0)
    }

    /// Clears retry state for a completed round.
    pub fn clear_round(&self, round_id: u64) {
        self.attempts.write().retain(|_, state| state.round_id != round_id);
    }
}

// ============================================================================
// TransactionRetryManager
// ============================================================================

/// Errors from transaction submission.
#[derive(Debug, Clone)]
pub enum TransactionError {
    /// Transaction reverted on-chain (retryable — may be nonce/gas issue).
    Reverted(String),
    /// Nonce too low (retryable — resubmit with fresh nonce).
    NonceTooLow(String),
    /// Gas estimation failed (retryable — network congestion).
    GasEstimationFailed(String),
    /// Insufficient balance (permanent — cannot retry).
    InsufficientBalance(String),
    /// Invalid proof rejected by contract (permanent — proof is wrong).
    InvalidProof(String),
    /// Network/RPC error (retryable — transient connectivity).
    NetworkError(String),
    /// Timeout waiting for confirmation.
    Timeout(String),
    /// All retries exhausted.
    RetriesExhausted {
        attempts: u32,
        last_error: String,
    },
}

impl std::fmt::Display for TransactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reverted(msg) => write!(f, "transaction reverted: {}", msg),
            Self::NonceTooLow(msg) => write!(f, "nonce too low: {}", msg),
            Self::GasEstimationFailed(msg) => write!(f, "gas estimation failed: {}", msg),
            Self::InsufficientBalance(msg) => write!(f, "insufficient balance: {}", msg),
            Self::InvalidProof(msg) => write!(f, "invalid proof: {}", msg),
            Self::NetworkError(msg) => write!(f, "network error: {}", msg),
            Self::Timeout(msg) => write!(f, "timeout: {}", msg),
            Self::RetriesExhausted { attempts, last_error } => {
                write!(f, "retries exhausted after {} attempts: {}", attempts, last_error)
            }
        }
    }
}

impl std::error::Error for TransactionError {}

impl TransactionError {
    /// Returns whether this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Reverted(_)
                | Self::NonceTooLow(_)
                | Self::GasEstimationFailed(_)
                | Self::NetworkError(_)
                | Self::Timeout(_)
        )
    }

    /// Classifies an error string from ethers/chain interaction.
    pub fn classify(error_msg: &str) -> Self {
        let lower = error_msg.to_lowercase();
        if lower.contains("nonce too low") || lower.contains("replacement transaction") {
            Self::NonceTooLow(error_msg.to_string())
        } else if lower.contains("insufficient funds") || lower.contains("insufficient balance") {
            Self::InsufficientBalance(error_msg.to_string())
        } else if lower.contains("gas") && lower.contains("estimation") {
            Self::GasEstimationFailed(error_msg.to_string())
        } else if lower.contains("invalid proof") || lower.contains("verification failed") {
            Self::InvalidProof(error_msg.to_string())
        } else if lower.contains("revert") || lower.contains("execution reverted") {
            Self::Reverted(error_msg.to_string())
        } else if lower.contains("timeout") || lower.contains("timed out") {
            Self::Timeout(error_msg.to_string())
        } else {
            Self::NetworkError(error_msg.to_string())
        }
    }
}

/// Manages chain transaction retries with exponential backoff.
pub struct TransactionRetryManager {
    policy: RetryPolicy,
    /// Total successful submissions.
    total_submitted: AtomicU64,
    /// Total failed submissions (after all retries exhausted).
    total_failed: AtomicU64,
}

impl TransactionRetryManager {
    /// Creates a new transaction retry manager.
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            total_submitted: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
        }
    }

    /// Creates a manager with default chain transaction policy.
    pub fn default_policy() -> Self {
        Self::new(RetryPolicy::for_chain_transaction())
    }

    /// Submits a transaction with automatic retry on retryable failures.
    ///
    /// The `operation` closure is called on each attempt. It should return
    /// `Ok(T)` on success or `Err(anyhow::Error)` on failure. The error
    /// message is classified to determine retryability.
    pub async fn submit_with_retry<F, Fut, T>(
        &self,
        operation_name: &str,
        mut operation: F,
    ) -> Result<T, TransactionError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, anyhow::Error>>,
    {
        let mut last_error = String::new();

        for attempt in 0..=self.policy.max_retries {
            if attempt > 0 {
                let delay = self.policy.delay_for_attempt(attempt - 1);
                info!(
                    operation = operation_name,
                    attempt,
                    delay_ms = delay.as_millis(),
                    "Retrying transaction after delay"
                );
                tokio::time::sleep(delay).await;
            }

            match operation().await {
                Ok(result) => {
                    self.total_submitted.fetch_add(1, Ordering::Relaxed);
                    if attempt > 0 {
                        info!(
                            operation = operation_name,
                            attempt,
                            "Transaction succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    let tx_error = TransactionError::classify(&e.to_string());
                    last_error = e.to_string();

                    if !tx_error.is_retryable() {
                        error!(
                            operation = operation_name,
                            attempt,
                            error = %e,
                            "Transaction failed with permanent error, not retrying"
                        );
                        self.total_failed.fetch_add(1, Ordering::Relaxed);
                        return Err(tx_error);
                    }

                    if attempt < self.policy.max_retries {
                        warn!(
                            operation = operation_name,
                            attempt,
                            max_retries = self.policy.max_retries,
                            error = %e,
                            "Transaction failed with retryable error"
                        );
                    } else {
                        error!(
                            operation = operation_name,
                            attempt,
                            error = %e,
                            "Transaction failed, no more retries"
                        );
                    }
                }
            }
        }

        self.total_failed.fetch_add(1, Ordering::Relaxed);
        Err(TransactionError::RetriesExhausted {
            attempts: self.policy.max_retries + 1,
            last_error,
        })
    }

    /// Returns submission statistics.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.total_submitted.load(Ordering::Relaxed),
            self.total_failed.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// WorkerFailureHandler
// ============================================================================

/// Action to take when a worker fails during a training round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureAction {
    /// Remove the worker from the current round and continue with remaining workers.
    RemoveFromRound {
        peer_id: PeerId,
        reason: String,
    },
    /// Continue in degraded mode (fewer workers than ideal but above minimum).
    ContinueDegraded {
        peer_id: PeerId,
        remaining_workers: usize,
    },
    /// Abort the current round (insufficient workers to continue).
    AbortRound {
        reason: String,
        remaining_workers: usize,
        minimum_required: usize,
    },
    /// No action needed (worker already handled or not in round).
    NoAction,
}

/// Handles worker failures during training rounds by deciding on recovery actions.
pub struct WorkerFailureHandler {
    /// Minimum workers required to continue a round.
    min_workers: usize,
}

impl WorkerFailureHandler {
    /// Creates a new worker failure handler.
    pub fn new(min_workers: usize) -> Self {
        Self { min_workers }
    }

    /// Evaluates what action to take when a worker fails.
    ///
    /// Decision logic:
    /// - During Announcing/WaitingForQuorum: just remove, hasn't started yet
    /// - During Distributing: remove and check if enough remain
    /// - During Training: remove, proceed with partial results if enough remain
    /// - During Aggregating/Submitting/Finalizing: too late to remove, continue degraded
    pub fn handle_failure(
        &self,
        peer_id: &PeerId,
        phase: &JobPhase,
        current_worker_count: usize,
        worker_in_round: bool,
    ) -> FailureAction {
        if !worker_in_round {
            return FailureAction::NoAction;
        }

        let remaining = current_worker_count.saturating_sub(1);

        match phase {
            // Early phases — safe to remove
            JobPhase::Idle | JobPhase::Announcing | JobPhase::WaitingForQuorum => {
                info!(
                    worker = %peer_id,
                    phase = %phase,
                    "Worker failed in early phase, removing"
                );
                FailureAction::RemoveFromRound {
                    peer_id: peer_id.clone(),
                    reason: format!("worker failed during {}", phase),
                }
            }

            // Distribution phase — remove if enough remain
            JobPhase::Distributing => {
                if remaining >= self.min_workers {
                    info!(
                        worker = %peer_id,
                        remaining,
                        min = self.min_workers,
                        "Worker failed during distribution, removing (enough workers remain)"
                    );
                    FailureAction::RemoveFromRound {
                        peer_id: peer_id.clone(),
                        reason: "worker failed during model distribution".to_string(),
                    }
                } else {
                    warn!(
                        worker = %peer_id,
                        remaining,
                        min = self.min_workers,
                        "Worker failed during distribution, aborting round"
                    );
                    FailureAction::AbortRound {
                        reason: format!(
                            "worker {} failed during distribution, only {}/{} workers remain",
                            peer_id, remaining, self.min_workers,
                        ),
                        remaining_workers: remaining,
                        minimum_required: self.min_workers,
                    }
                }
            }

            // Training phase — remove and check if enough proofs possible
            JobPhase::Training => {
                if remaining >= self.min_workers {
                    info!(
                        worker = %peer_id,
                        remaining,
                        min = self.min_workers,
                        "Worker failed during training, continuing with remaining"
                    );
                    FailureAction::RemoveFromRound {
                        peer_id: peer_id.clone(),
                        reason: "worker failed during training".to_string(),
                    }
                } else {
                    warn!(
                        worker = %peer_id,
                        remaining,
                        min = self.min_workers,
                        "Worker failed during training, aborting round"
                    );
                    FailureAction::AbortRound {
                        reason: format!(
                            "worker {} failed during training, only {}/{} workers remain",
                            peer_id, remaining, self.min_workers,
                        ),
                        remaining_workers: remaining,
                        minimum_required: self.min_workers,
                    }
                }
            }

            // Late phases — too late to remove, just note degradation
            JobPhase::Aggregating | JobPhase::Submitting | JobPhase::Finalizing => {
                warn!(
                    worker = %peer_id,
                    phase = %phase,
                    "Worker failed in late phase, continuing degraded"
                );
                FailureAction::ContinueDegraded {
                    peer_id: peer_id.clone(),
                    remaining_workers: remaining,
                }
            }

            // Terminal phases — no action needed
            JobPhase::Complete | JobPhase::Failed => FailureAction::NoAction,
        }
    }
}

// ============================================================================
// RoundTimeoutManager
// ============================================================================

/// Tracks timeout state for a single phase of a training round.
#[derive(Debug)]
pub struct RoundTimeoutManager {
    /// When the current phase started.
    phase_start: Instant,
    /// Configured timeout for the current phase.
    phase_timeout: Duration,
    /// Current phase being tracked.
    current_phase: JobPhase,
}

impl RoundTimeoutManager {
    /// Creates a new timeout manager in idle state.
    pub fn new() -> Self {
        Self {
            phase_start: Instant::now(),
            phase_timeout: Duration::from_secs(u64::MAX),
            current_phase: JobPhase::Idle,
        }
    }

    /// Starts tracking a new phase with the given timeout.
    pub fn start_phase(&mut self, phase: JobPhase, timeout: Duration) {
        self.phase_start = Instant::now();
        self.phase_timeout = timeout;
        self.current_phase = phase;
    }

    /// Returns whether the current phase has expired.
    pub fn is_expired(&self) -> bool {
        self.phase_start.elapsed() >= self.phase_timeout
    }

    /// Returns the remaining time before timeout.
    pub fn remaining(&self) -> Duration {
        self.phase_timeout
            .checked_sub(self.phase_start.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Returns the elapsed time since phase start.
    pub fn elapsed(&self) -> Duration {
        self.phase_start.elapsed()
    }

    /// Returns whether we can proceed with partial results.
    ///
    /// True if the timeout has been reached AND we have at least the minimum
    /// required results to continue.
    pub fn can_proceed_partial(&self, received: usize, minimum: usize) -> bool {
        self.is_expired() && received >= minimum
    }

    /// Returns the current phase.
    pub fn current_phase(&self) -> &JobPhase {
        &self.current_phase
    }
}

impl Default for RoundTimeoutManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GracefulShutdownCoordinator
// ============================================================================

/// Coordinates graceful shutdown when SIGTERM/SIGINT is received.
///
/// When a shutdown signal arrives:
/// 1. Stops accepting new rounds
/// 2. Waits for the active round to complete (if any)
/// 3. Saves aggregator checkpoint
/// 4. Exits cleanly
pub struct GracefulShutdownCoordinator {
    /// Whether shutdown has been requested.
    shutdown_requested: Arc<AtomicBool>,
    /// Currently active round ID (0 = no active round).
    active_round: Arc<AtomicU64>,
    /// Notifier for shutdown completion.
    shutdown_complete: Arc<tokio::sync::Notify>,
}

impl GracefulShutdownCoordinator {
    /// Creates a new graceful shutdown coordinator.
    pub fn new() -> Self {
        Self {
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            active_round: Arc::new(AtomicU64::new(0)),
            shutdown_complete: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Installs signal handlers for SIGTERM and SIGINT.
    ///
    /// Returns a future that resolves when a shutdown signal is received.
    /// The coordinator marks `shutdown_requested` and waits for the active
    /// round to complete before notifying `shutdown_complete`.
    pub fn install_signal_handlers(&self) -> tokio::task::JoinHandle<()> {
        let shutdown_requested = self.shutdown_requested.clone();
        let active_round = self.active_round.clone();
        let shutdown_complete = self.shutdown_complete.clone();

        tokio::spawn(async move {
            // Wait for either SIGTERM or SIGINT
            let ctrl_c = tokio::signal::ctrl_c();

            #[cfg(unix)]
            let terminate = async {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let terminate = std::future::pending::<()>();

            tokio::select! {
                _ = ctrl_c => {
                    info!("Received SIGINT, initiating graceful shutdown");
                }
                _ = terminate => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                }
            }

            shutdown_requested.store(true, Ordering::SeqCst);

            // Wait for active round to complete
            let round_id = active_round.load(Ordering::SeqCst);
            if round_id > 0 {
                info!(
                    round_id,
                    "Waiting for active round to complete before shutdown"
                );

                // Poll until active round clears
                loop {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    if active_round.load(Ordering::SeqCst) == 0 {
                        break;
                    }
                }

                info!(round_id, "Active round completed, proceeding with shutdown");
            }

            shutdown_complete.notify_waiters();
        })
    }

    /// Returns whether shutdown has been requested.
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    /// Registers that a training round is actively in progress.
    pub fn register_active_round(&self, round_id: u64) {
        self.active_round.store(round_id, Ordering::SeqCst);
    }

    /// Clears the active round (call when round completes or fails).
    pub fn clear_active_round(&self) {
        self.active_round.store(0, Ordering::SeqCst);
    }

    /// Returns whether new rounds should be accepted.
    ///
    /// Returns false once shutdown has been requested. The aggregator
    /// should stop starting new rounds but allow the current one to finish.
    pub fn should_accept_new_round(&self) -> bool {
        !self.shutdown_requested()
    }

    /// Returns the currently active round ID (0 if none).
    pub fn active_round_id(&self) -> u64 {
        self.active_round.load(Ordering::SeqCst)
    }

    /// Waits for the shutdown process to complete.
    ///
    /// This resolves after:
    /// 1. A shutdown signal is received
    /// 2. The active round (if any) finishes
    pub async fn wait_for_shutdown(&self) {
        self.shutdown_complete.notified().await;
    }
}

impl Default for GracefulShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FaultRecoveryConfig
// ============================================================================

/// Combined configuration for all fault recovery mechanisms.
#[derive(Debug, Clone)]
pub struct FaultRecoveryConfig {
    /// Retry policy for proof generation.
    pub proof_retry_policy: RetryPolicy,
    /// Retry policy for chain transactions.
    pub tx_retry_policy: RetryPolicy,
    /// Minimum workers required to continue a round.
    pub min_workers: usize,
    /// Whether to enable graceful shutdown handling.
    pub graceful_shutdown: bool,
}

impl Default for FaultRecoveryConfig {
    fn default() -> Self {
        Self {
            proof_retry_policy: RetryPolicy::for_proof_generation(),
            tx_retry_policy: RetryPolicy::for_chain_transaction(),
            min_workers: 1,
            graceful_shutdown: true,
        }
    }
}

impl FaultRecoveryConfig {
    /// Creates config from node-level settings.
    pub fn from_min_workers(min_workers: usize) -> Self {
        Self {
            min_workers,
            ..Default::default()
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- RetryPolicy tests ---

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert!(policy.should_retry(0));
        assert!(policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_factor: 2.0,
        };

        let d0 = policy.delay_for_attempt(0);
        let d1 = policy.delay_for_attempt(1);
        let d2 = policy.delay_for_attempt(2);

        // Delays should generally increase (accounting for jitter)
        // Base: 100, 200, 400 ms with ±25% jitter
        assert!(d0.as_millis() >= 50 && d0.as_millis() <= 200);
        assert!(d1.as_millis() >= 100 && d1.as_millis() <= 400);
        assert!(d2.as_millis() >= 200 && d2.as_millis() <= 800);
    }

    #[test]
    fn test_retry_policy_max_delay_cap() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_factor: 10.0,
        };

        // After enough attempts, delay should be capped
        let d = policy.delay_for_attempt(5);
        assert!(d <= Duration::from_millis(7500)); // 5s * 1.25 jitter max
    }

    // --- ProofRetryManager tests ---

    #[test]
    fn test_proof_retry_manager_basic() {
        let mgr = ProofRetryManager::new(RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            backoff_factor: 2.0,
        });

        let worker = PeerId::from_string("worker-1");
        let round_id = 1;

        // Initially should retry (no attempts yet)
        assert!(mgr.should_retry(&worker, round_id));
        assert_eq!(mgr.attempt_count(&worker, round_id), 0);

        // First attempt
        mgr.register_attempt(&worker, round_id);
        assert_eq!(mgr.attempt_count(&worker, round_id), 1);
        mgr.record_failure(&worker, round_id, "circuit error".to_string());
        assert!(mgr.should_retry(&worker, round_id));

        // Second attempt
        mgr.register_attempt(&worker, round_id);
        assert_eq!(mgr.attempt_count(&worker, round_id), 2);
        mgr.record_failure(&worker, round_id, "circuit error".to_string());
        // max_retries=2, we've had 2 attempts, should not retry
        assert!(!mgr.should_retry(&worker, round_id));

        // Should appear in permanently failed
        let failed = mgr.permanently_failed(round_id);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], worker);
    }

    #[test]
    fn test_proof_retry_manager_success_stops_retries() {
        let mgr = ProofRetryManager::new(RetryPolicy {
            max_retries: 5,
            ..RetryPolicy::default()
        });

        let worker = PeerId::from_string("worker-1");
        let round_id = 1;

        mgr.register_attempt(&worker, round_id);
        mgr.record_failure(&worker, round_id, "error".to_string());

        mgr.register_attempt(&worker, round_id);
        mgr.record_success(&worker, round_id);

        // Should not retry after success
        assert!(!mgr.should_retry(&worker, round_id));
        // Should not appear in permanently failed
        assert!(mgr.permanently_failed(round_id).is_empty());
    }

    #[test]
    fn test_proof_retry_manager_round_reset() {
        let mgr = ProofRetryManager::new(RetryPolicy {
            max_retries: 1,
            ..RetryPolicy::default()
        });

        let worker = PeerId::from_string("worker-1");

        // Round 1: exhaust retries
        mgr.register_attempt(&worker, 1);
        mgr.record_failure(&worker, 1, "error".to_string());
        assert!(!mgr.should_retry(&worker, 1));

        // Round 2: should be able to retry (fresh state)
        mgr.register_attempt(&worker, 2);
        assert_eq!(mgr.attempt_count(&worker, 2), 1);

        // Old round should be cleared
        mgr.clear_round(1);
        assert!(mgr.permanently_failed(1).is_empty());
    }

    #[test]
    fn test_proof_retry_manager_multiple_workers() {
        let mgr = ProofRetryManager::new(RetryPolicy {
            max_retries: 1,
            ..RetryPolicy::default()
        });

        let w1 = PeerId::from_string("worker-1");
        let w2 = PeerId::from_string("worker-2");
        let w3 = PeerId::from_string("worker-3");
        let round_id = 1;

        // w1: fails permanently
        mgr.register_attempt(&w1, round_id);
        mgr.record_failure(&w1, round_id, "error".to_string());

        // w2: succeeds
        mgr.register_attempt(&w2, round_id);
        mgr.record_success(&w2, round_id);

        // w3: hasn't tried
        let failed = mgr.permanently_failed(round_id);
        assert_eq!(failed.len(), 1);
        assert!(failed.contains(&w1));
    }

    // --- TransactionError tests ---

    #[test]
    fn test_transaction_error_classification() {
        assert!(TransactionError::classify("nonce too low").is_retryable());
        assert!(TransactionError::classify("execution reverted: 0x1234").is_retryable());
        assert!(TransactionError::classify("gas estimation failed").is_retryable());
        assert!(TransactionError::classify("connection timed out").is_retryable());
        assert!(TransactionError::classify("network error: ECONNREFUSED").is_retryable());

        assert!(!TransactionError::classify("insufficient funds for transfer").is_retryable());
        assert!(!TransactionError::classify("invalid proof: verification failed").is_retryable());
    }

    // --- TransactionRetryManager tests ---

    #[tokio::test]
    async fn test_transaction_retry_success_first_try() {
        let mgr = TransactionRetryManager::new(RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
        });

        let result = mgr.submit_with_retry("test_op", || async {
            Ok::<_, anyhow::Error>("tx_hash_123".to_string())
        }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "tx_hash_123");
        assert_eq!(mgr.stats(), (1, 0));
    }

    #[tokio::test]
    async fn test_transaction_retry_success_after_retries() {
        let mgr = TransactionRetryManager::new(RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
        });

        let attempt_count = Arc::new(AtomicU64::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = mgr.submit_with_retry("test_op", move || {
            let count = attempt_count_clone.fetch_add(1, Ordering::SeqCst);
            async move {
                if count < 2 {
                    Err(anyhow::anyhow!("connection timed out"))
                } else {
                    Ok("tx_hash_456".to_string())
                }
            }
        }).await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
        assert_eq!(mgr.stats(), (1, 0));
    }

    #[tokio::test]
    async fn test_transaction_retry_permanent_failure() {
        let mgr = TransactionRetryManager::new(RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
        });

        let result: Result<String, _> = mgr.submit_with_retry("test_op", || async {
            Err::<String, _>(anyhow::anyhow!("insufficient funds for transfer"))
        }).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            TransactionError::InsufficientBalance(_) => {}
            other => panic!("expected InsufficientBalance, got {:?}", other),
        }
        assert_eq!(mgr.stats(), (0, 1));
    }

    #[tokio::test]
    async fn test_transaction_retry_exhausted() {
        let mgr = TransactionRetryManager::new(RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            backoff_factor: 2.0,
        });

        let attempt_count = Arc::new(AtomicU64::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result: Result<String, _> = mgr.submit_with_retry("test_op", move || {
            attempt_count_clone.fetch_add(1, Ordering::SeqCst);
            async { Err::<String, _>(anyhow::anyhow!("execution reverted: 0xdead")) }
        }).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            TransactionError::RetriesExhausted { attempts, .. } => {
                assert_eq!(attempts, 3); // initial + 2 retries
            }
            other => panic!("expected RetriesExhausted, got {:?}", other),
        }
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    // --- WorkerFailureHandler tests ---

    #[test]
    fn test_worker_failure_handler_early_phase() {
        let handler = WorkerFailureHandler::new(2);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::WaitingForQuorum, 5, true);
        match action {
            FailureAction::RemoveFromRound { peer_id, .. } => {
                assert_eq!(peer_id, worker);
            }
            other => panic!("expected RemoveFromRound, got {:?}", other),
        }
    }

    #[test]
    fn test_worker_failure_handler_training_enough_workers() {
        let handler = WorkerFailureHandler::new(2);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::Training, 3, true);
        match action {
            FailureAction::RemoveFromRound { peer_id, .. } => {
                assert_eq!(peer_id, worker);
            }
            other => panic!("expected RemoveFromRound, got {:?}", other),
        }
    }

    #[test]
    fn test_worker_failure_handler_training_insufficient_workers() {
        let handler = WorkerFailureHandler::new(3);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::Training, 3, true);
        match action {
            FailureAction::AbortRound { remaining_workers, minimum_required, .. } => {
                assert_eq!(remaining_workers, 2);
                assert_eq!(minimum_required, 3);
            }
            other => panic!("expected AbortRound, got {:?}", other),
        }
    }

    #[test]
    fn test_worker_failure_handler_late_phase_degraded() {
        let handler = WorkerFailureHandler::new(2);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::Aggregating, 3, true);
        match action {
            FailureAction::ContinueDegraded { peer_id, remaining_workers } => {
                assert_eq!(peer_id, worker);
                assert_eq!(remaining_workers, 2);
            }
            other => panic!("expected ContinueDegraded, got {:?}", other),
        }
    }

    #[test]
    fn test_worker_failure_handler_not_in_round() {
        let handler = WorkerFailureHandler::new(2);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::Training, 3, false);
        assert_eq!(action, FailureAction::NoAction);
    }

    #[test]
    fn test_worker_failure_handler_complete_phase() {
        let handler = WorkerFailureHandler::new(2);
        let worker = PeerId::from_string("worker-1");

        let action = handler.handle_failure(&worker, &JobPhase::Complete, 3, true);
        assert_eq!(action, FailureAction::NoAction);
    }

    // --- RoundTimeoutManager tests ---

    #[test]
    fn test_round_timeout_manager_basic() {
        let mut mgr = RoundTimeoutManager::new();

        mgr.start_phase(JobPhase::Training, Duration::from_secs(300));
        assert!(!mgr.is_expired());
        assert!(mgr.remaining() > Duration::ZERO);
        assert_eq!(*mgr.current_phase(), JobPhase::Training);
    }

    #[test]
    fn test_round_timeout_manager_expired() {
        let mut mgr = RoundTimeoutManager::new();

        mgr.start_phase(JobPhase::Training, Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert!(mgr.is_expired());
        assert_eq!(mgr.remaining(), Duration::ZERO);
    }

    #[test]
    fn test_round_timeout_manager_partial_proceed() {
        let mut mgr = RoundTimeoutManager::new();

        mgr.start_phase(JobPhase::Training, Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));

        // Expired with enough results → can proceed
        assert!(mgr.can_proceed_partial(3, 2));
        // Expired without enough results → cannot proceed
        assert!(!mgr.can_proceed_partial(1, 2));
    }

    #[test]
    fn test_round_timeout_manager_not_expired_cannot_proceed() {
        let mut mgr = RoundTimeoutManager::new();

        mgr.start_phase(JobPhase::Training, Duration::from_secs(300));

        // Not expired → can_proceed_partial is false even with enough results
        assert!(!mgr.can_proceed_partial(5, 2));
    }

    // --- GracefulShutdownCoordinator tests ---

    #[test]
    fn test_graceful_shutdown_initial_state() {
        let coord = GracefulShutdownCoordinator::new();
        assert!(!coord.shutdown_requested());
        assert!(coord.should_accept_new_round());
        assert_eq!(coord.active_round_id(), 0);
    }

    #[test]
    fn test_graceful_shutdown_active_round_tracking() {
        let coord = GracefulShutdownCoordinator::new();

        coord.register_active_round(42);
        assert_eq!(coord.active_round_id(), 42);

        coord.clear_active_round();
        assert_eq!(coord.active_round_id(), 0);
    }

    #[test]
    fn test_graceful_shutdown_stops_new_rounds() {
        let coord = GracefulShutdownCoordinator::new();

        // Simulate shutdown request
        coord.shutdown_requested.store(true, Ordering::SeqCst);

        assert!(coord.shutdown_requested());
        assert!(!coord.should_accept_new_round());
    }

    #[tokio::test]
    async fn test_graceful_shutdown_waits_for_active_round() {
        let coord = GracefulShutdownCoordinator::new();

        // Register an active round
        coord.register_active_round(1);

        // Request shutdown
        coord.shutdown_requested.store(true, Ordering::SeqCst);

        // Spawn a task that clears the round after a short delay
        let coord_clone_active = coord.active_round.clone();
        let coord_clone_notify = coord.shutdown_complete.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            coord_clone_active.store(0, Ordering::SeqCst);
            // In the real signal handler, shutdown_complete is notified
            // after the active round is cleared. Simulate that here.
            coord_clone_notify.notify_waiters();
        });

        // Wait for shutdown should complete after the round clears
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            coord.wait_for_shutdown(),
        ).await;

        assert!(result.is_ok());
        assert_eq!(coord.active_round_id(), 0);
    }

    // --- Integrated scenario tests ---

    #[test]
    fn test_worker_crash_mid_training_with_enough_workers() {
        // Simulate: 5 workers in round, 1 crashes, min_workers=3
        let handler = WorkerFailureHandler::new(3);

        let crashed = PeerId::from_string("worker-2");
        let action = handler.handle_failure(
            &crashed,
            &JobPhase::Training,
            5,  // current count before removal
            true,
        );

        // Should remove and continue (5-1=4 >= 3)
        match action {
            FailureAction::RemoveFromRound { peer_id, reason } => {
                assert_eq!(peer_id, crashed);
                assert!(reason.contains("training"));
            }
            other => panic!("expected RemoveFromRound, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_workers_crash_cascading() {
        let handler = WorkerFailureHandler::new(3);

        // First crash: 5→4 workers, OK
        let action1 = handler.handle_failure(
            &PeerId::from_string("w1"),
            &JobPhase::Training,
            5,
            true,
        );
        assert!(matches!(action1, FailureAction::RemoveFromRound { .. }));

        // Second crash: 4→3 workers, still OK
        let action2 = handler.handle_failure(
            &PeerId::from_string("w2"),
            &JobPhase::Training,
            4,
            true,
        );
        assert!(matches!(action2, FailureAction::RemoveFromRound { .. }));

        // Third crash: 3→2 workers, below minimum → abort
        let action3 = handler.handle_failure(
            &PeerId::from_string("w3"),
            &JobPhase::Training,
            3,
            true,
        );
        assert!(matches!(action3, FailureAction::AbortRound { .. }));
    }

    #[test]
    fn test_proof_retry_with_worker_failure() {
        // Scenario: Worker's proof fails, retries work
        let retry_mgr = ProofRetryManager::new(RetryPolicy {
            max_retries: 3,
            ..RetryPolicy::default()
        });

        let worker = PeerId::from_string("worker-1");
        let round_id = 1;

        // First attempt fails
        retry_mgr.register_attempt(&worker, round_id);
        retry_mgr.record_failure(&worker, round_id, "OOM".to_string());
        assert!(retry_mgr.should_retry(&worker, round_id));

        // Second attempt fails
        retry_mgr.register_attempt(&worker, round_id);
        retry_mgr.record_failure(&worker, round_id, "OOM".to_string());
        assert!(retry_mgr.should_retry(&worker, round_id));

        // Third attempt succeeds
        retry_mgr.register_attempt(&worker, round_id);
        retry_mgr.record_success(&worker, round_id);
        assert!(!retry_mgr.should_retry(&worker, round_id));
        assert!(retry_mgr.permanently_failed(round_id).is_empty());
    }
}
