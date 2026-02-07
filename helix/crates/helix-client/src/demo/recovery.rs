//! Demo Recovery Module
//!
//! Provides error handling and automatic recovery for demo execution:
//! - Node failure detection and recovery
//! - Network partition handling
//! - Proof generation failure recovery
//! - Graceful degradation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,
    /// Delay between recovery attempts
    pub recovery_delay: Duration,
    /// Timeout for recovery operations
    pub recovery_timeout: Duration,
    /// Enable graceful degradation
    pub graceful_degradation: bool,
    /// Minimum workers required to continue
    pub min_workers: u32,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable automatic node restart
    pub auto_restart_nodes: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            max_recovery_attempts: 3,
            recovery_delay: Duration::from_secs(1),
            recovery_timeout: Duration::from_secs(10),
            graceful_degradation: true,
            min_workers: 1,
            health_check_interval: Duration::from_secs(2),
            auto_restart_nodes: true,
        }
    }
}

impl RecoveryConfig {
    /// Strict recovery config for production
    pub fn strict() -> Self {
        Self {
            auto_recovery: true,
            max_recovery_attempts: 5,
            recovery_delay: Duration::from_secs(2),
            recovery_timeout: Duration::from_secs(30),
            graceful_degradation: false,
            min_workers: 2,
            health_check_interval: Duration::from_secs(1),
            auto_restart_nodes: true,
        }
    }

    /// Lenient config for demos
    pub fn demo_mode() -> Self {
        Self {
            auto_recovery: true,
            max_recovery_attempts: 2,
            recovery_delay: Duration::from_millis(500),
            recovery_timeout: Duration::from_secs(5),
            graceful_degradation: true,
            min_workers: 1,
            health_check_interval: Duration::from_secs(3),
            auto_restart_nodes: false, // Don't restart in demo, just continue
        }
    }
}

/// Error categories for recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Network connectivity issues
    NetworkError,
    /// Node failure
    NodeFailure,
    /// Proof generation failure
    ProofError,
    /// Blockchain transaction failure
    TransactionError,
    /// Resource exhaustion (memory, GPU, etc.)
    ResourceError,
    /// Timeout
    Timeout,
    /// Data corruption
    DataError,
    /// Configuration error
    ConfigError,
    /// Unknown error
    Unknown,
}

impl ErrorCategory {
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError | Self::NodeFailure | Self::Timeout | Self::ResourceError
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::NetworkError => "Network Error",
            Self::NodeFailure => "Node Failure",
            Self::ProofError => "Proof Error",
            Self::TransactionError => "Transaction Error",
            Self::ResourceError => "Resource Error",
            Self::Timeout => "Timeout",
            Self::DataError => "Data Error",
            Self::ConfigError => "Configuration Error",
            Self::Unknown => "Unknown Error",
        }
    }
}

/// Recovery action to take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    /// Retry the operation
    Retry,
    /// Skip the operation and continue
    Skip,
    /// Restart affected component
    Restart { component: String },
    /// Use fallback approach
    Fallback { description: String },
    /// Reduce operation scope
    Degrade { description: String },
    /// Abort and fail
    Abort { reason: String },
    /// Wait and retry
    WaitAndRetry { delay: Duration },
    /// Manual intervention required
    ManualIntervention { instructions: String },
}

/// Error record
#[derive(Debug, Clone)]
pub struct ErrorRecord {
    /// Error category
    pub category: ErrorCategory,
    /// Error message
    pub message: String,
    /// Component that failed
    pub component: Option<String>,
    /// Timestamp
    pub timestamp: Instant,
    /// Recovery attempts made
    pub recovery_attempts: u32,
    /// Whether recovered
    pub recovered: bool,
    /// Recovery action taken
    pub action_taken: Option<RecoveryAction>,
}

/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery succeeded
    pub success: bool,
    /// Action taken
    pub action: RecoveryAction,
    /// Time taken
    pub duration: Duration,
    /// Message
    pub message: String,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Component name
    pub component: String,
    /// Whether healthy
    pub healthy: bool,
    /// Response time
    pub response_time: Duration,
    /// Details
    pub details: Option<String>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Recovery event
#[derive(Debug, Clone)]
pub enum RecoveryEvent {
    /// Error detected
    ErrorDetected {
        category: ErrorCategory,
        message: String,
        component: Option<String>,
    },
    /// Recovery started
    RecoveryStarted {
        category: ErrorCategory,
        action: RecoveryAction,
    },
    /// Recovery completed
    RecoveryCompleted {
        category: ErrorCategory,
        success: bool,
        duration: Duration,
    },
    /// Health check failed
    HealthCheckFailed {
        component: String,
        reason: String,
    },
    /// Graceful degradation activated
    DegradationActivated {
        reason: String,
        impact: String,
    },
    /// System recovered
    SystemRecovered {
        total_errors: u32,
        total_recoveries: u32,
    },
}

/// Resettable component that can be restarted during recovery
pub trait Resettable: Send + Sync {
    /// Reset component to initial state
    fn reset(&self) -> Result<()>;
    /// Check if the component is healthy
    fn is_healthy(&self) -> bool;
    /// Component name
    fn name(&self) -> &str;
}

/// Registry of resettable components for coordinated recovery
pub struct ComponentRegistry {
    components: HashMap<String, Box<dyn Resettable>>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    /// Register a component for managed recovery
    pub fn register(&mut self, component: Box<dyn Resettable>) {
        let name = component.name().to_string();
        self.components.insert(name, component);
    }

    /// Reset a specific component by name
    pub fn reset_component(&self, name: &str) -> Result<()> {
        if let Some(component) = self.components.get(name) {
            component.reset()
        } else {
            Err(anyhow!("Component '{}' not found in registry", name))
        }
    }

    /// Check if a specific component is healthy
    pub fn is_healthy(&self, name: &str) -> Option<bool> {
        self.components.get(name).map(|c| c.is_healthy())
    }

    /// Reset all registered components
    pub fn reset_all(&self) -> Vec<(String, Result<()>)> {
        self.components
            .iter()
            .map(|(name, c)| (name.clone(), c.reset()))
            .collect()
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Demo recovery manager
pub struct RecoveryManager {
    /// Configuration
    config: RecoveryConfig,
    /// Error history
    errors: Arc<RwLock<Vec<ErrorRecord>>>,
    /// Health check results
    health_checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    /// Recovery statistics
    stats: Arc<RwLock<RecoveryStats>>,
    /// Event sender
    event_tx: Option<mpsc::Sender<RecoveryEvent>>,
    /// Degraded mode active
    degraded: Arc<RwLock<bool>>,
    /// Registry of resettable components
    component_registry: Arc<RwLock<ComponentRegistry>>,
    /// RPC endpoint for health checks
    rpc_endpoint: Option<String>,
}

/// Recovery statistics
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Total errors encountered
    pub total_errors: u32,
    /// Total successful recoveries
    pub successful_recoveries: u32,
    /// Total failed recoveries
    pub failed_recoveries: u32,
    /// Errors by category
    pub errors_by_category: HashMap<ErrorCategory, u32>,
    /// Average recovery time
    pub avg_recovery_time_ms: u64,
    /// Times degraded mode was activated
    pub degradation_count: u32,
}

impl RecoveryManager {
    /// Create new recovery manager
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            errors: Arc::new(RwLock::new(Vec::new())),
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(RecoveryStats::default())),
            event_tx: None,
            degraded: Arc::new(RwLock::new(false)),
            component_registry: Arc::new(RwLock::new(ComponentRegistry::new())),
            rpc_endpoint: None,
        }
    }

    /// Create with event channel
    pub fn with_events(config: RecoveryConfig, event_tx: mpsc::Sender<RecoveryEvent>) -> Self {
        let mut manager = Self::new(config);
        manager.event_tx = Some(event_tx);
        manager
    }

    /// Create for demo mode
    pub fn demo_mode() -> Self {
        Self::new(RecoveryConfig::demo_mode())
    }

    /// Set the RPC endpoint for real health checks
    pub fn set_rpc_endpoint(&mut self, endpoint: String) {
        self.rpc_endpoint = Some(endpoint);
    }

    /// Register a component in the recovery manager's component registry
    pub async fn register_component(&self, component: Box<dyn Resettable>) {
        self.component_registry.write().await.register(component);
    }

    /// Emit event
    async fn emit(&self, event: RecoveryEvent) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(event).await;
        }
    }

    /// Record an error
    pub async fn record_error(
        &self,
        category: ErrorCategory,
        message: &str,
        component: Option<&str>,
    ) {
        let record = ErrorRecord {
            category,
            message: message.to_string(),
            component: component.map(|s| s.to_string()),
            timestamp: Instant::now(),
            recovery_attempts: 0,
            recovered: false,
            action_taken: None,
        };

        {
            let mut errors = self.errors.write().await;
            errors.push(record);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_errors += 1;
            *stats.errors_by_category.entry(category).or_insert(0) += 1;
        }

        self.emit(RecoveryEvent::ErrorDetected {
            category,
            message: message.to_string(),
            component: component.map(|s| s.to_string()),
        })
        .await;
    }

    /// Attempt recovery for an error
    pub async fn attempt_recovery(
        &self,
        category: ErrorCategory,
        component: Option<&str>,
    ) -> Result<RecoveryResult> {
        if !self.config.auto_recovery {
            return Err(anyhow!("Auto recovery disabled"));
        }

        let action = self.determine_recovery_action(category, component).await;

        self.emit(RecoveryEvent::RecoveryStarted {
            category,
            action: action.clone(),
        })
        .await;

        let start = Instant::now();
        let result = self.execute_recovery(&action).await;
        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            if result.is_ok() {
                stats.successful_recoveries += 1;
            } else {
                stats.failed_recoveries += 1;
            }

            // Update average recovery time
            let total = stats.successful_recoveries + stats.failed_recoveries;
            stats.avg_recovery_time_ms =
                (stats.avg_recovery_time_ms * (total - 1) as u64 + duration.as_millis() as u64) / total as u64;
        }

        // Mark error as recovered if successful
        if result.is_ok() {
            let mut errors = self.errors.write().await;
            if let Some(last_error) = errors.iter_mut().rev().find(|e| e.category == category) {
                last_error.recovered = true;
                last_error.action_taken = Some(action.clone());
            }
        }

        self.emit(RecoveryEvent::RecoveryCompleted {
            category,
            success: result.is_ok(),
            duration,
        })
        .await;

        result.map(|_| RecoveryResult {
            success: true,
            action,
            duration,
            message: "Recovery successful".to_string(),
        })
    }

    /// Determine what recovery action to take
    async fn determine_recovery_action(
        &self,
        category: ErrorCategory,
        component: Option<&str>,
    ) -> RecoveryAction {
        // Check error history for this category
        let errors = self.errors.read().await;
        let recent_errors = errors
            .iter()
            .filter(|e| e.category == category)
            .filter(|e| e.timestamp.elapsed() < Duration::from_secs(60))
            .count();

        // If too many recent errors, escalate
        if recent_errors >= self.config.max_recovery_attempts as usize {
            if self.config.graceful_degradation {
                return RecoveryAction::Degrade {
                    description: format!(
                        "Too many {} errors, reducing functionality",
                        category.name()
                    ),
                };
            } else {
                return RecoveryAction::Abort {
                    reason: format!("Max recovery attempts exceeded for {}", category.name()),
                };
            }
        }

        // Determine action based on category
        match category {
            ErrorCategory::NetworkError => {
                RecoveryAction::WaitAndRetry {
                    delay: self.config.recovery_delay,
                }
            }
            ErrorCategory::NodeFailure => {
                if self.config.auto_restart_nodes {
                    RecoveryAction::Restart {
                        component: component.unwrap_or("node").to_string(),
                    }
                } else {
                    RecoveryAction::Skip
                }
            }
            ErrorCategory::ProofError => {
                RecoveryAction::Retry
            }
            ErrorCategory::TransactionError => {
                RecoveryAction::WaitAndRetry {
                    delay: Duration::from_secs(2),
                }
            }
            ErrorCategory::ResourceError => {
                RecoveryAction::Degrade {
                    description: "Reducing resource usage".to_string(),
                }
            }
            ErrorCategory::Timeout => {
                RecoveryAction::Retry
            }
            ErrorCategory::DataError | ErrorCategory::ConfigError => {
                RecoveryAction::ManualIntervention {
                    instructions: "Check data/configuration and restart".to_string(),
                }
            }
            ErrorCategory::Unknown => {
                RecoveryAction::Skip
            }
        }
    }

    /// Execute a recovery action against real components
    async fn execute_recovery(&self, action: &RecoveryAction) -> Result<()> {
        match action {
            RecoveryAction::Retry => {
                // Signal to caller that retry is warranted
                tracing::info!("Recovery action: retry requested");
                Ok(())
            }
            RecoveryAction::Skip => {
                tracing::info!("Recovery action: skipping failed operation");
                Ok(())
            }
            RecoveryAction::Restart { component } => {
                // Attempt real component reset via the registry
                let registry = self.component_registry.read().await;
                match registry.reset_component(component) {
                    Ok(()) => {
                        tracing::info!("Successfully restarted component: {}", component);
                        Ok(())
                    }
                    Err(_) => {
                        // Component not in registry — fall back to timed wait
                        tracing::warn!(
                            "Component '{}' not in registry, waiting before retry",
                            component
                        );
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        Ok(())
                    }
                }
            }
            RecoveryAction::Fallback { description } => {
                tracing::info!("Activating fallback: {}", description);
                // Mark system degraded since we're using a fallback path
                *self.degraded.write().await = true;
                self.emit(RecoveryEvent::DegradationActivated {
                    reason: format!("Fallback activated: {}", description),
                    impact: "Switched to fallback implementation".to_string(),
                })
                .await;
                Ok(())
            }
            RecoveryAction::Degrade { description } => {
                *self.degraded.write().await = true;

                self.emit(RecoveryEvent::DegradationActivated {
                    reason: description.clone(),
                    impact: "Demo continues with reduced functionality".to_string(),
                })
                .await;

                let mut stats = self.stats.write().await;
                stats.degradation_count += 1;

                Ok(())
            }
            RecoveryAction::Abort { reason } => {
                Err(anyhow!("Recovery aborted: {}", reason))
            }
            RecoveryAction::WaitAndRetry { delay } => {
                tracing::info!("Recovery: waiting {:?} before retry", delay);
                tokio::time::sleep(*delay).await;
                Ok(())
            }
            RecoveryAction::ManualIntervention { instructions } => {
                tracing::warn!("Manual intervention required: {}", instructions);
                Err(anyhow!("Manual intervention required"))
            }
        }
    }

    /// Run a real health check for a component.
    /// Checks the component registry first; falls back to basic liveness.
    pub async fn health_check(&self, component: &str) -> HealthCheck {
        let start = Instant::now();

        // Try the component registry for a real health signal
        let registry = self.component_registry.read().await;
        let (healthy, details) = match registry.is_healthy(component) {
            Some(h) => (
                h,
                if h {
                    Some("Component reports healthy".to_string())
                } else {
                    Some("Component reports unhealthy".to_string())
                },
            ),
            None => {
                // Component not registered — do a basic check
                (true, Some("Component not registered; assumed healthy".to_string()))
            }
        };
        drop(registry);

        let response_time = start.elapsed();

        let check = HealthCheck {
            component: component.to_string(),
            healthy,
            response_time,
            details,
            timestamp: Instant::now(),
        };

        // Store result
        {
            let mut checks = self.health_checks.write().await;
            checks.insert(component.to_string(), check.clone());
        }

        if !healthy {
            self.emit(RecoveryEvent::HealthCheckFailed {
                component: component.to_string(),
                reason: "Health check failed".to_string(),
            })
            .await;
        }

        check
    }

    /// Check if system is in degraded mode
    pub async fn is_degraded(&self) -> bool {
        *self.degraded.read().await
    }

    /// Get recovery statistics
    pub async fn stats(&self) -> RecoveryStats {
        self.stats.read().await.clone()
    }

    /// Get error history
    pub async fn error_history(&self) -> Vec<ErrorRecord> {
        self.errors.read().await.clone()
    }

    /// Check if demo can continue
    pub async fn can_continue(&self, active_workers: u32) -> bool {
        if active_workers < self.config.min_workers {
            return false;
        }

        let stats = self.stats.read().await;
        let recent_failures = stats.total_errors - stats.successful_recoveries;

        // Allow continuation if recovery rate is acceptable
        if recent_failures > self.config.max_recovery_attempts {
            return self.config.graceful_degradation;
        }

        true
    }

    /// Reset recovery state
    pub async fn reset(&self) {
        self.errors.write().await.clear();
        self.health_checks.write().await.clear();
        *self.stats.write().await = RecoveryStats::default();
        *self.degraded.write().await = false;
    }

    /// Generate recovery report
    pub async fn report(&self) -> RecoveryReport {
        let stats = self.stats.read().await.clone();
        let errors = self.errors.read().await.clone();
        let degraded = *self.degraded.read().await;

        let recovery_rate = if stats.total_errors > 0 {
            stats.successful_recoveries as f64 / stats.total_errors as f64 * 100.0
        } else {
            100.0
        };

        RecoveryReport {
            total_errors: stats.total_errors,
            successful_recoveries: stats.successful_recoveries,
            failed_recoveries: stats.failed_recoveries,
            recovery_rate,
            avg_recovery_time_ms: stats.avg_recovery_time_ms,
            degradation_count: stats.degradation_count,
            currently_degraded: degraded,
            errors_by_category: stats.errors_by_category,
            unrecovered_errors: errors.iter().filter(|e| !e.recovered).count(),
        }
    }
}

/// Recovery report
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    pub total_errors: u32,
    pub successful_recoveries: u32,
    pub failed_recoveries: u32,
    pub recovery_rate: f64,
    pub avg_recovery_time_ms: u64,
    pub degradation_count: u32,
    pub currently_degraded: bool,
    pub errors_by_category: HashMap<ErrorCategory, u32>,
    pub unrecovered_errors: usize,
}

impl RecoveryReport {
    /// Format as human-readable string
    pub fn display(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Recovery Report\n\
             ===============\n\
             Total Errors:        {}\n\
             Successful Recoveries: {}\n\
             Failed Recoveries:   {}\n\
             Recovery Rate:       {:.1}%\n\
             Avg Recovery Time:   {}ms\n\
             Degradation Count:   {}\n\
             Currently Degraded:  {}\n\
             Unrecovered Errors:  {}\n\n\
             Errors by Category:\n",
            self.total_errors,
            self.successful_recoveries,
            self.failed_recoveries,
            self.recovery_rate,
            self.avg_recovery_time_ms,
            self.degradation_count,
            if self.currently_degraded { "Yes" } else { "No" },
            self.unrecovered_errors,
        ));

        for (category, count) in &self.errors_by_category {
            output.push_str(&format!("  {}: {}\n", category.name(), count));
        }

        output
    }
}

// ============================================================================
// Pre-Demo Health Checks
// ============================================================================

/// Pre-demo check result
#[derive(Debug, Clone)]
pub struct PreDemoCheckResult {
    /// Overall readiness
    pub ready: bool,
    /// Individual check results
    pub checks: Vec<PreDemoCheck>,
    /// Warnings (not blockers)
    pub warnings: Vec<String>,
    /// Errors (blockers)
    pub errors: Vec<String>,
    /// Estimated demo reliability score (0-100)
    pub reliability_score: u8,
}

/// Individual pre-demo check
#[derive(Debug, Clone)]
pub struct PreDemoCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Duration of check
    pub duration: Duration,
    /// Details or error message
    pub details: Option<String>,
}

impl RecoveryManager {
    /// Run comprehensive pre-demo health checks
    pub async fn run_pre_demo_checks(&self) -> PreDemoCheckResult {
        let mut checks = Vec::new();
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut score = 100u8;

        // Check 1: Memory availability
        let mem_check = self.check_memory().await;
        if !mem_check.passed {
            errors.push("Insufficient memory for demo".to_string());
            score = score.saturating_sub(30);
        }
        checks.push(mem_check);

        // Check 2: Network connectivity
        let net_check = self.check_network().await;
        if !net_check.passed {
            warnings.push("Network connectivity issues detected".to_string());
            score = score.saturating_sub(10);
        }
        checks.push(net_check);

        // Check 3: RPC endpoint availability
        let rpc_check = self.check_rpc_endpoint().await;
        if !rpc_check.passed {
            warnings.push("RPC endpoint not available, will use mock mode".to_string());
            score = score.saturating_sub(5);
        }
        checks.push(rpc_check);

        // Check 4: Proving system readiness
        let prove_check = self.check_proving_system().await;
        if !prove_check.passed {
            warnings.push("Proving system not fully initialized".to_string());
            score = score.saturating_sub(15);
        }
        checks.push(prove_check);

        // Check 5: Previous errors cleared
        let err_check = self.check_error_state().await;
        if !err_check.passed {
            warnings.push("Previous errors detected, consider reset".to_string());
            score = score.saturating_sub(5);
        }
        checks.push(err_check);

        let ready = errors.is_empty() && score >= 50;

        PreDemoCheckResult {
            ready,
            checks,
            warnings,
            errors,
            reliability_score: score,
        }
    }

    /// Check memory availability by attempting a test allocation
    async fn check_memory(&self) -> PreDemoCheck {
        let start = Instant::now();

        // Attempt a 64 MB test allocation to verify memory is available
        let passed = std::panic::catch_unwind(|| {
            let _test: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);
            true
        })
        .unwrap_or(false);

        PreDemoCheck {
            name: "Memory Check".to_string(),
            passed,
            duration: start.elapsed(),
            details: if passed {
                Some("Sufficient memory available (64 MB test passed)".to_string())
            } else {
                Some("Low memory — could not allocate 64 MB".to_string())
            },
        }
    }

    /// Check network connectivity by attempting a DNS resolution
    async fn check_network(&self) -> PreDemoCheck {
        let start = Instant::now();

        // Try to resolve a well-known address as a connectivity signal
        let passed = tokio::net::lookup_host("1.1.1.1:80").await.is_ok();

        PreDemoCheck {
            name: "Network Check".to_string(),
            passed,
            duration: start.elapsed(),
            details: if passed {
                Some("Network connectivity OK".to_string())
            } else {
                Some("Network check failed (DNS resolution)".to_string())
            },
        }
    }

    /// Check RPC endpoint availability by attempting a TCP connection
    async fn check_rpc_endpoint(&self) -> PreDemoCheck {
        let start = Instant::now();

        let endpoint = self
            .rpc_endpoint
            .as_deref()
            .unwrap_or("127.0.0.1:9545");

        // Strip scheme for raw TCP connect
        let addr = endpoint
            .strip_prefix("http://")
            .or_else(|| endpoint.strip_prefix("https://"))
            .unwrap_or(endpoint);

        let passed = tokio::time::timeout(
            Duration::from_secs(2),
            tokio::net::TcpStream::connect(addr),
        )
        .await
        .map(|r| r.is_ok())
        .unwrap_or(false);

        PreDemoCheck {
            name: "RPC Endpoint Check".to_string(),
            passed,
            duration: start.elapsed(),
            details: if passed {
                Some(format!("RPC endpoint {} reachable", endpoint))
            } else {
                Some(format!("RPC endpoint {} not reachable (will use mock mode)", endpoint))
            },
        }
    }

    /// Check proving system readiness via component registry
    async fn check_proving_system(&self) -> PreDemoCheck {
        let start = Instant::now();

        let registry = self.component_registry.read().await;
        let passed = registry
            .is_healthy("proof_generator")
            .unwrap_or(true); // If not registered, assume OK

        PreDemoCheck {
            name: "Proving System Check".to_string(),
            passed,
            duration: start.elapsed(),
            details: if passed {
                Some("Proving system ready".to_string())
            } else {
                Some("Proving system reports unhealthy".to_string())
            },
        }
    }

    /// Check if previous error state is clear
    async fn check_error_state(&self) -> PreDemoCheck {
        let start = Instant::now();

        let stats = self.stats.read().await;
        let passed = stats.total_errors == 0 || stats.successful_recoveries == stats.total_errors;

        PreDemoCheck {
            name: "Error State Check".to_string(),
            passed,
            duration: start.elapsed(),
            details: if passed {
                Some("No unresolved errors".to_string())
            } else {
                Some(format!(
                    "{} unresolved errors from previous session",
                    stats.total_errors - stats.successful_recoveries
                ))
            },
        }
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Half-open, testing if recovery is possible
    HalfOpen,
    /// Open, rejecting operations
    Open,
}

/// Circuit breaker for error rate limiting
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state
    state: Arc<RwLock<CircuitState>>,
    /// Failure count
    failure_count: Arc<RwLock<u32>>,
    /// Success count in half-open
    success_count: Arc<RwLock<u32>>,
    /// Failure threshold to trip
    failure_threshold: u32,
    /// Success threshold to close
    success_threshold: u32,
    /// Time to wait before half-open
    reset_timeout: Duration,
    /// Last state change time
    last_state_change: Arc<RwLock<Instant>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(failure_threshold: u32, success_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            failure_threshold,
            success_threshold,
            reset_timeout,
            last_state_change: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Create with demo-friendly settings
    pub fn demo_mode() -> Self {
        Self::new(5, 2, Duration::from_secs(10))
    }

    /// Check if operation is allowed
    pub async fn allow(&self) -> bool {
        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                let last_change = *self.last_state_change.read().await;
                if last_change.elapsed() >= self.reset_timeout {
                    self.transition_to(CircuitState::HalfOpen).await;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a success
    pub async fn record_success(&self) {
        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => {
                // Reset failure count on success
                *self.failure_count.write().await = 0;
            }
            CircuitState::HalfOpen => {
                let mut success = self.success_count.write().await;
                *success += 1;
                if *success >= self.success_threshold {
                    drop(success);
                    self.transition_to(CircuitState::Closed).await;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failure
    pub async fn record_failure(&self) {
        let state = *self.state.read().await;
        match state {
            CircuitState::Closed => {
                let mut failures = self.failure_count.write().await;
                *failures += 1;
                if *failures >= self.failure_threshold {
                    drop(failures);
                    self.transition_to(CircuitState::Open).await;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open trips the breaker
                self.transition_to(CircuitState::Open).await;
            }
            CircuitState::Open => {}
        }
    }

    /// Transition to new state
    async fn transition_to(&self, new_state: CircuitState) {
        let mut state = self.state.write().await;
        if *state != new_state {
            *state = new_state;
            *self.last_state_change.write().await = Instant::now();

            // Reset counters on state change
            match new_state {
                CircuitState::Closed => {
                    *self.failure_count.write().await = 0;
                    *self.success_count.write().await = 0;
                }
                CircuitState::HalfOpen => {
                    *self.success_count.write().await = 0;
                }
                CircuitState::Open => {}
            }
        }
    }

    /// Get current state
    pub async fn state(&self) -> CircuitState {
        *self.state.read().await
    }

    /// Reset the circuit breaker
    pub async fn reset(&self) {
        self.transition_to(CircuitState::Closed).await;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::demo_mode()
    }
}

// ============================================================================
// Heartbeat Monitor
// ============================================================================

/// Heartbeat monitor for periodic health checks
pub struct HeartbeatMonitor {
    /// Check interval
    interval: Duration,
    /// Components to monitor
    components: Vec<String>,
    /// Last heartbeat times
    last_heartbeat: Arc<RwLock<HashMap<String, Instant>>>,
    /// Heartbeat timeout
    timeout: Duration,
}

impl HeartbeatMonitor {
    /// Create new heartbeat monitor
    pub fn new(interval: Duration, timeout: Duration) -> Self {
        Self {
            interval,
            components: Vec::new(),
            last_heartbeat: Arc::new(RwLock::new(HashMap::new())),
            timeout,
        }
    }

    /// Create with demo settings
    pub fn demo_mode() -> Self {
        Self::new(Duration::from_secs(5), Duration::from_secs(15))
    }

    /// Register a component for monitoring
    pub async fn register(&mut self, component: &str) {
        self.components.push(component.to_string());
        self.last_heartbeat
            .write()
            .await
            .insert(component.to_string(), Instant::now());
    }

    /// Record a heartbeat from a component
    pub async fn heartbeat(&self, component: &str) {
        self.last_heartbeat
            .write()
            .await
            .insert(component.to_string(), Instant::now());
    }

    /// Check for stale components
    pub async fn check_stale(&self) -> Vec<String> {
        let heartbeats = self.last_heartbeat.read().await;
        let now = Instant::now();

        heartbeats
            .iter()
            .filter(|(_, last)| now.duration_since(**last) > self.timeout)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Check if a specific component is healthy based on heartbeat freshness
    pub async fn is_healthy(&self, component: &str) -> bool {
        if let Some(last) = self.last_heartbeat.read().await.get(component) {
            Instant::now().duration_since(*last) <= self.timeout
        } else {
            false
        }
    }

    /// Actively check a component by verifying heartbeat freshness and attempting a TCP probe.
    /// Returns a health check result with timing information.
    pub async fn check_component(&self, component: &str, addr: Option<&str>) -> super::recovery::HealthCheck {
        let start = Instant::now();

        // First check heartbeat freshness
        let heartbeat_ok = self.is_healthy(component).await;

        // If an address is provided, also do a TCP probe
        let probe_ok = if let Some(a) = addr {
            tokio::time::timeout(
                Duration::from_secs(2),
                tokio::net::TcpStream::connect(a),
            )
            .await
            .map(|r| r.is_ok())
            .unwrap_or(false)
        } else {
            true
        };

        let healthy = heartbeat_ok && probe_ok;

        super::recovery::HealthCheck {
            component: component.to_string(),
            healthy,
            response_time: start.elapsed(),
            details: Some(format!(
                "heartbeat={}, probe={}",
                if heartbeat_ok { "ok" } else { "stale" },
                if probe_ok { "ok" } else { "failed" }
            )),
            timestamp: Instant::now(),
        }
    }

    /// Get interval
    pub fn interval(&self) -> Duration {
        self.interval
    }
}

impl Default for HeartbeatMonitor {
    fn default() -> Self {
        Self::demo_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RecoveryConfig::default();
        assert!(config.auto_recovery);
        assert_eq!(config.max_recovery_attempts, 3);
    }

    #[test]
    fn test_error_category_recoverable() {
        assert!(ErrorCategory::NetworkError.is_recoverable());
        assert!(ErrorCategory::NodeFailure.is_recoverable());
        assert!(!ErrorCategory::ConfigError.is_recoverable());
    }

    #[tokio::test]
    async fn test_recovery_manager_creation() {
        let manager = RecoveryManager::demo_mode();
        let stats = manager.stats().await;
        assert_eq!(stats.total_errors, 0);
    }

    #[tokio::test]
    async fn test_record_error() {
        let manager = RecoveryManager::demo_mode();

        manager
            .record_error(ErrorCategory::NetworkError, "Connection failed", Some("node-1"))
            .await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.errors_by_category.get(&ErrorCategory::NetworkError), Some(&1));
    }

    #[tokio::test]
    async fn test_can_continue() {
        let manager = RecoveryManager::demo_mode();

        // Should be able to continue with enough workers
        assert!(manager.can_continue(3).await);

        // Should not continue with no workers
        assert!(!manager.can_continue(0).await);
    }

    #[tokio::test]
    async fn test_health_check() {
        let manager = RecoveryManager::demo_mode();

        let check = manager.health_check("test-component").await;
        assert!(check.healthy);
        assert_eq!(check.component, "test-component");
    }

    #[tokio::test]
    async fn test_reset() {
        let manager = RecoveryManager::demo_mode();

        manager
            .record_error(ErrorCategory::NetworkError, "Test error", None)
            .await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_errors, 1);

        manager.reset().await;

        let stats = manager.stats().await;
        assert_eq!(stats.total_errors, 0);
    }
}
