//! Circuit breaker pattern for external service calls.
//!
//! Wraps calls to Ethereum RPC, IPFS, and S3 with automatic failure detection
//! and recovery. When a service exceeds its failure threshold, the circuit opens
//! and rejects requests immediately until a reset timeout expires, then allows
//! limited probe requests (half-open) before fully closing again.

use std::future::Future;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// CircuitState
// ---------------------------------------------------------------------------

/// Circuit breaker states following the standard three-state model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation -- requests pass through.
    Closed = 0,
    /// Failures exceeded threshold -- requests rejected immediately.
    Open = 1,
    /// Testing recovery -- limited requests allowed.
    HalfOpen = 2,
}

impl CircuitState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Closed,
            1 => Self::Open,
            2 => Self::HalfOpen,
            _ => Self::Closed, // defensive fallback
        }
    }
}

// ---------------------------------------------------------------------------
// CircuitBreakerError
// ---------------------------------------------------------------------------

/// Error type returned by [`CircuitBreaker::call`].
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError<E: std::fmt::Debug> {
    /// The circuit breaker is open and rejecting requests.
    #[error("circuit breaker '{name}' is open, rejecting request")]
    CircuitOpen {
        /// Name of the circuit breaker that rejected the request.
        name: String,
    },
    /// The underlying service call failed.
    #[error("service error: {0:?}")]
    ServiceError(E),
}

// ---------------------------------------------------------------------------
// CircuitBreakerConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,
    /// Duration to wait in the Open state before transitioning to HalfOpen.
    pub reset_timeout: Duration,
    /// Number of consecutive successful requests in HalfOpen required to close.
    pub success_threshold: u32,
    /// Human-readable name for structured logging.
    pub name: String,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            success_threshold: 2,
            name: "unnamed".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker
// ---------------------------------------------------------------------------

/// A circuit breaker that wraps external service calls with automatic failure
/// detection, fast-fail behaviour, and gradual recovery.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    /// Encoded state: 0 = Closed, 1 = Open, 2 = HalfOpen.
    state: AtomicU8,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        info!(
            name = %config.name,
            failure_threshold = config.failure_threshold,
            reset_timeout_ms = config.reset_timeout.as_millis() as u64,
            "circuit breaker created"
        );
        Self {
            config,
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: Mutex::new(None),
        }
    }

    /// Returns the current [`CircuitState`], automatically transitioning from
    /// Open to HalfOpen if the reset timeout has elapsed.
    pub fn state(&self) -> CircuitState {
        let raw = self.state.load(Ordering::SeqCst);
        let current = CircuitState::from_u8(raw);

        if current == CircuitState::Open {
            let last_failure = self.last_failure_time.lock();
            if let Some(t) = *last_failure {
                if t.elapsed() >= self.config.reset_timeout {
                    drop(last_failure); // release lock before CAS
                    // Attempt transition Open -> HalfOpen.
                    if self
                        .state
                        .compare_exchange(
                            CircuitState::Open as u8,
                            CircuitState::HalfOpen as u8,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                    {
                        self.success_count.store(0, Ordering::Relaxed);
                        info!(
                            name = %self.config.name,
                            "circuit breaker transitioning Open -> HalfOpen"
                        );
                        return CircuitState::HalfOpen;
                    }
                }
            }
        }

        current
    }

    /// Returns `true` if a request should be allowed through based on the
    /// current state.
    ///
    /// - **Closed**: always allows.
    /// - **HalfOpen**: always allows (probe request).
    /// - **Open**: rejects unless the reset timeout has elapsed (which triggers
    ///   an automatic transition to HalfOpen).
    pub fn allow_request(&self) -> bool {
        match self.state() {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    /// Record a successful call. In the HalfOpen state, once
    /// `success_threshold` consecutive successes are recorded the circuit
    /// transitions back to Closed.
    pub fn record_success(&self) {
        let current = self.state();

        match current {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                debug!(
                    name = %self.config.name,
                    count,
                    threshold = self.config.success_threshold,
                    "half-open success recorded"
                );
                if count >= self.config.success_threshold as u64 {
                    self.state
                        .store(CircuitState::Closed as u8, Ordering::SeqCst);
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                    info!(
                        name = %self.config.name,
                        "circuit breaker closed after successful recovery"
                    );
                }
            }
            CircuitState::Closed => {
                // Reset failure count on any success in Closed state.
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {
                // Should not normally happen (requests are rejected), but
                // handle gracefully.
            }
        }
    }

    /// Record a failed call. In the Closed state, once `failure_threshold`
    /// consecutive failures are recorded the circuit opens. In the HalfOpen
    /// state, any failure immediately reopens the circuit.
    pub fn record_failure(&self) {
        let current = self.state();

        // Update last failure time.
        {
            let mut lft = self.last_failure_time.lock();
            *lft = Some(Instant::now());
        }

        match current {
            CircuitState::Closed => {
                let count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                warn!(
                    name = %self.config.name,
                    count,
                    threshold = self.config.failure_threshold,
                    "failure recorded in closed state"
                );
                if count >= self.config.failure_threshold as u64 {
                    self.state
                        .store(CircuitState::Open as u8, Ordering::SeqCst);
                    warn!(
                        name = %self.config.name,
                        "circuit breaker OPENED after {} consecutive failures",
                        count
                    );
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in HalfOpen immediately reopens.
                self.state
                    .store(CircuitState::Open as u8, Ordering::SeqCst);
                self.success_count.store(0, Ordering::Relaxed);
                warn!(
                    name = %self.config.name,
                    "circuit breaker re-opened from HalfOpen after failure"
                );
            }
            CircuitState::Open => {
                // Already open; just update timestamp.
            }
        }
    }

    /// Manually reset the circuit breaker to the Closed state.
    pub fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::SeqCst);
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        {
            let mut lft = self.last_failure_time.lock();
            *lft = None;
        }
        info!(name = %self.config.name, "circuit breaker manually reset to Closed");
    }

    /// Wrap an async call with circuit breaker logic.
    ///
    /// - If the circuit is **Open**, returns [`CircuitBreakerError::CircuitOpen`]
    ///   immediately without executing `f`.
    /// - Otherwise, executes `f`. On success, records a success and returns
    ///   `Ok(T)`. On failure, records a failure and returns
    ///   [`CircuitBreakerError::ServiceError`].
    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        if !self.allow_request() {
            return Err(CircuitBreakerError::CircuitOpen {
                name: self.config.name.clone(),
            });
        }

        match f.await {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(e) => {
                self.record_failure();
                Err(CircuitBreakerError::ServiceError(e))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ServiceBreakers -- pre-configured breakers for all external services
// ---------------------------------------------------------------------------

/// Pre-configured circuit breakers for all external services used by the node.
pub struct ServiceBreakers {
    /// Ethereum JSON-RPC calls.
    pub chain_rpc: CircuitBreaker,
    /// IPFS HTTP gateway / API calls.
    pub ipfs: CircuitBreaker,
    /// S3-compatible object storage calls.
    pub s3: CircuitBreaker,
}

impl ServiceBreakers {
    /// Create a new set of service breakers with sensible defaults.
    pub fn new() -> Self {
        Self {
            chain_rpc: CircuitBreaker::new(CircuitBreakerConfig {
                failure_threshold: 3,
                reset_timeout: Duration::from_secs(15),
                success_threshold: 1,
                name: "chain_rpc".to_string(),
            }),
            ipfs: CircuitBreaker::new(CircuitBreakerConfig {
                failure_threshold: 5,
                reset_timeout: Duration::from_secs(30),
                success_threshold: 2,
                name: "ipfs".to_string(),
            }),
            s3: CircuitBreaker::new(CircuitBreakerConfig {
                failure_threshold: 5,
                reset_timeout: Duration::from_secs(30),
                success_threshold: 2,
                name: "s3".to_string(),
            }),
        }
    }
}

impl Default for ServiceBreakers {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(name: &str) -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_millis(50),
            success_threshold: 2,
            name: name.to_string(),
        }
    }

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(test_config("test_closed"));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_opens_after_failures() {
        let cb = CircuitBreaker::new(test_config("test_open"));

        // Record failures up to threshold.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure(); // 3rd failure = threshold
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_open_rejects_requests() {
        let cb = CircuitBreaker::new(test_config("test_reject"));

        // Force open.
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_transitions_to_half_open() {
        let cb = CircuitBreaker::new(test_config("test_half_open"));

        // Open the circuit.
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for reset timeout.
        std::thread::sleep(Duration::from_millis(60));

        // state() should detect the elapsed timeout and transition.
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_closes_after_success_in_half_open() {
        let cb = CircuitBreaker::new(test_config("test_close_half"));

        // Open the circuit.
        for _ in 0..3 {
            cb.record_failure();
        }

        // Wait for reset timeout -> HalfOpen.
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Record successes up to threshold (2).
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_reopens_on_half_open_failure() {
        let cb = CircuitBreaker::new(test_config("test_reopen"));

        // Open the circuit.
        for _ in 0..3 {
            cb.record_failure();
        }

        // Wait for reset timeout -> HalfOpen.
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // A failure in HalfOpen should immediately reopen.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }

    #[tokio::test]
    async fn test_call_wraps_errors() {
        let cb = CircuitBreaker::new(test_config("test_call_err"));

        // Open the circuit.
        for _ in 0..3 {
            cb.record_failure();
        }

        // call() should return CircuitOpen without executing the future.
        let result: Result<(), CircuitBreakerError<String>> =
            cb.call(async { Ok::<(), String>(()) }).await;

        match result {
            Err(CircuitBreakerError::CircuitOpen { name }) => {
                assert_eq!(name, "test_call_err");
            }
            other => panic!("expected CircuitOpen, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_call_records_success() {
        let cb = CircuitBreaker::new(test_config("test_call_ok"));

        let result: Result<u32, CircuitBreakerError<String>> =
            cb.call(async { Ok::<u32, String>(42) }).await;

        assert_eq!(result.unwrap(), 42);
        // After a success, failure count should be 0.
        assert_eq!(cb.failure_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_reset() {
        let cb = CircuitBreaker::new(test_config("test_reset"));

        // Open the circuit.
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);

        // Manual reset.
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.allow_request());
        assert_eq!(cb.failure_count.load(Ordering::Relaxed), 0);
        assert_eq!(cb.success_count.load(Ordering::Relaxed), 0);
    }
}
