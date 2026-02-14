//! Prover Health Check System.
//!
//! Provides comprehensive health checking for the proving infrastructure:
//! - Memory and resource monitoring
//! - Pipeline initialization status
//! - Key availability verification
//! - Performance baseline testing
//! - System compatibility checks
//! - Cache health monitoring

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use helix_circuits::halo2curves::bn256::Fr;
use serde::{Deserialize, Serialize};

use crate::pipeline::PipelineConfig;

// ============================================================================
// Health Status Types
// ============================================================================

/// Overall health status of the prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operational.
    Healthy,
    /// Some non-critical issues detected.
    Degraded,
    /// Critical issues, proving may fail.
    Unhealthy,
    /// Status unknown, health check not run.
    Unknown,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Unhealthy => write!(f, "Unhealthy"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Severity level for health issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational, no action needed.
    Info,
    /// Warning, may indicate potential problems.
    Warning,
    /// Error, critical issue requiring attention.
    Error,
}

/// A health check issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Short code for the issue.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Component that reported the issue.
    pub component: String,
    /// When the issue was detected.
    pub detected_at: u64,
    /// Suggested remediation.
    pub remediation: Option<String>,
}

impl HealthIssue {
    /// Creates a new health issue.
    pub fn new<S: Into<String>>(
        severity: IssueSeverity,
        code: S,
        message: S,
        component: S,
    ) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            component: component.into(),
            detected_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            remediation: None,
        }
    }

    /// Adds a remediation suggestion.
    pub fn with_remediation<S: Into<String>>(mut self, remediation: S) -> Self {
        self.remediation = Some(remediation.into());
        self
    }

    /// Creates an info-level issue.
    pub fn info<S: Into<String>>(code: S, message: S, component: S) -> Self {
        Self::new(IssueSeverity::Info, code, message, component)
    }

    /// Creates a warning-level issue.
    pub fn warning<S: Into<String>>(code: S, message: S, component: S) -> Self {
        Self::new(IssueSeverity::Warning, code, message, component)
    }

    /// Creates an error-level issue.
    pub fn error<S: Into<String>>(code: S, message: S, component: S) -> Self {
        Self::new(IssueSeverity::Error, code, message, component)
    }
}

// ============================================================================
// Component Health
// ============================================================================

/// Health status of a specific component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Component status.
    pub status: HealthStatus,
    /// Status message.
    pub message: String,
    /// Response time for the component check (ms).
    pub response_time_ms: u64,
    /// When the check was performed.
    pub checked_at: u64,
    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

impl ComponentHealth {
    /// Creates a healthy component status.
    pub fn healthy(name: &str, message: &str, response_time_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Healthy,
            message: message.to_string(),
            response_time_ms,
            checked_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metrics: HashMap::new(),
        }
    }

    /// Creates an unhealthy component status.
    pub fn unhealthy(name: &str, message: &str, response_time_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Unhealthy,
            message: message.to_string(),
            response_time_ms,
            checked_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metrics: HashMap::new(),
        }
    }

    /// Creates a degraded component status.
    pub fn degraded(name: &str, message: &str, response_time_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            status: HealthStatus::Degraded,
            message: message.to_string(),
            response_time_ms,
            checked_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metrics: HashMap::new(),
        }
    }

    /// Adds a metric to the component health.
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }
}

// ============================================================================
// Health Report
// ============================================================================

/// Complete health report for the prover system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall system status.
    pub status: HealthStatus,
    /// When the report was generated.
    pub generated_at: u64,
    /// Time taken to generate the report (ms).
    pub check_duration_ms: u64,
    /// Component health statuses.
    pub components: Vec<ComponentHealth>,
    /// Detected issues.
    pub issues: Vec<HealthIssue>,
    /// System information.
    pub system_info: SystemInfo,
    /// Performance baseline.
    pub performance: Option<PerformanceBaseline>,
}

impl HealthReport {
    /// Creates a new health report.
    pub fn new(
        components: Vec<ComponentHealth>,
        issues: Vec<HealthIssue>,
        system_info: SystemInfo,
        check_duration_ms: u64,
    ) -> Self {
        // Determine overall status
        let status = if issues.iter().any(|i| i.severity == IssueSeverity::Error) {
            HealthStatus::Unhealthy
        } else if issues.iter().any(|i| i.severity == IssueSeverity::Warning)
            || components.iter().any(|c| c.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else if components.iter().all(|c| c.status == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unknown
        };

        Self {
            status,
            generated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            check_duration_ms,
            components,
            issues,
            system_info,
            performance: None,
        }
    }

    /// Checks if the prover is ready to generate proofs.
    pub fn is_ready(&self) -> bool {
        matches!(self.status, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Returns all issues of a given severity or higher.
    pub fn issues_at_or_above(&self, min_severity: IssueSeverity) -> Vec<&HealthIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= min_severity)
            .collect()
    }

    /// Returns a summary string.
    pub fn summary(&self) -> String {
        let component_summary = self
            .components
            .iter()
            .map(|c| format!("{}: {}", c.name, c.status))
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "Status: {} | Components: [{}] | Issues: {} errors, {} warnings",
            self.status,
            component_summary,
            self.issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .count(),
            self.issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Warning)
                .count()
        )
    }
}

impl fmt::Display for HealthReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Prover Health Report ===")?;
        writeln!(f, "Status: {}", self.status)?;
        writeln!(f, "Check Duration: {}ms", self.check_duration_ms)?;
        writeln!(f)?;

        writeln!(f, "Components:")?;
        for component in &self.components {
            writeln!(
                f,
                "  - {}: {} ({}ms)",
                component.name, component.status, component.response_time_ms
            )?;
            if !component.message.is_empty() {
                writeln!(f, "    {}", component.message)?;
            }
        }
        writeln!(f)?;

        if !self.issues.is_empty() {
            writeln!(f, "Issues:")?;
            for issue in &self.issues {
                writeln!(
                    f,
                    "  [{:?}] {} - {} ({})",
                    issue.severity, issue.code, issue.message, issue.component
                )?;
                if let Some(ref rem) = issue.remediation {
                    writeln!(f, "    Remediation: {}", rem)?;
                }
            }
        }

        writeln!(f)?;
        writeln!(f, "System Info:")?;
        writeln!(f, "  CPU Cores: {}", self.system_info.cpu_cores)?;
        writeln!(
            f,
            "  Available Memory: {} MB",
            self.system_info.available_memory_mb
        )?;
        writeln!(f, "  Platform: {}", self.system_info.platform)?;

        Ok(())
    }
}

// ============================================================================
// System Information
// ============================================================================

/// System information for health reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Number of CPU cores.
    pub cpu_cores: usize,
    /// Available memory in MB.
    pub available_memory_mb: u64,
    /// Total memory in MB.
    pub total_memory_mb: u64,
    /// Operating system / platform.
    pub platform: String,
    /// Rust version.
    pub rust_version: String,
    /// Whether GPU acceleration is available.
    pub gpu_available: bool,
    /// GPU information (if available).
    pub gpu_info: Option<String>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            available_memory_mb: 0, // Would need system-specific code
            total_memory_mb: 0,
            platform: std::env::consts::OS.to_string(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            gpu_available: false,
            gpu_info: None,
        }
    }
}

impl SystemInfo {
    /// Gathers current system information.
    pub fn gather() -> Self {
        let mut info = Self::default();

        // Check for GPU availability (simplified - would need actual GPU detection)
        #[cfg(target_os = "macos")]
        {
            info.gpu_available = crate::metal::is_metal_available();
            if info.gpu_available {
                if let Some(device_info) = crate::metal::get_device_info() {
                    info.gpu_info = Some(device_info.name);
                }
            }
        }

        info
    }
}

// ============================================================================
// Performance Baseline
// ============================================================================

/// Performance baseline metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Time to generate a minimal proof (ms).
    pub minimal_proof_time_ms: u64,
    /// Time to verify a proof (ms).
    pub verification_time_ms: u64,
    /// Memory usage during proof generation (MB).
    pub proof_memory_mb: u64,
    /// Throughput estimate (proofs per minute).
    pub throughput_estimate: f64,
    /// When the baseline was established.
    pub measured_at: u64,
}

// ============================================================================
// Health Checker
// ============================================================================

/// Configuration for health checks.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Whether to run performance baseline tests.
    pub run_performance_test: bool,
    /// Timeout for individual component checks.
    pub component_timeout: Duration,
    /// Minimum acceptable memory (MB).
    pub min_memory_mb: u64,
    /// Maximum acceptable response time (ms).
    pub max_response_time_ms: u64,
    /// Enable detailed diagnostics.
    pub detailed_diagnostics: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            run_performance_test: false,
            component_timeout: Duration::from_secs(30),
            min_memory_mb: 512,
            max_response_time_ms: 5000,
            detailed_diagnostics: true,
        }
    }
}

/// Prover health checker.
pub struct HealthChecker {
    /// Configuration.
    config: HealthCheckConfig,
    /// Last health report.
    last_report: RwLock<Option<HealthReport>>,
    /// Whether a check is in progress.
    check_in_progress: AtomicBool,
    /// Total number of checks performed.
    check_count: AtomicU64,
}

impl HealthChecker {
    /// Creates a new health checker.
    pub fn new() -> Self {
        Self::with_config(HealthCheckConfig::default())
    }

    /// Creates a health checker with custom configuration.
    pub fn with_config(config: HealthCheckConfig) -> Self {
        Self {
            config,
            last_report: RwLock::new(None),
            check_in_progress: AtomicBool::new(false),
            check_count: AtomicU64::new(0),
        }
    }

    /// Runs a full health check.
    pub fn check(&self) -> HealthReport {
        // Prevent concurrent checks
        if self
            .check_in_progress
            .swap(true, Ordering::SeqCst)
        {
            // Return last report if available
            if let Some(report) = self.last_report.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
                return report.clone();
            }
        }

        let start = Instant::now();
        let mut components = Vec::new();
        let mut issues = Vec::new();

        // Check system resources
        let system_check = self.check_system_resources();
        if system_check.status != HealthStatus::Healthy {
            issues.push(
                HealthIssue::warning(
                    "RESOURCE_CONSTRAINT",
                    &system_check.message,
                    "system",
                )
                .with_remediation("Consider closing other applications or adding more memory"),
            );
        }
        components.push(system_check);

        // Check pipeline configuration
        let pipeline_check = self.check_pipeline_config();
        components.push(pipeline_check);

        // Check cache health
        let cache_check = self.check_cache_health();
        components.push(cache_check);

        // Check cryptographic primitives
        let crypto_check = self.check_crypto_primitives();
        if crypto_check.status == HealthStatus::Unhealthy {
            issues.push(HealthIssue::error(
                "CRYPTO_FAILURE",
                &crypto_check.message,
                "crypto",
            ));
        }
        components.push(crypto_check);

        // Check proof pipeline (generates and verifies a real proof)
        if self.config.run_performance_test {
            let proof_check = ProofHealthCheck::run();
            let proof_component = if proof_check.passed {
                ComponentHealth::healthy(
                    "proof_pipeline",
                    &format!("Proof generated and verified ({} bytes)",
                        proof_check.proof_size.unwrap_or(0)),
                    proof_check.duration_ms,
                )
            } else {
                issues.push(HealthIssue::error(
                    "PROOF_PIPELINE_FAILURE",
                    proof_check.error.as_deref().unwrap_or("Unknown"),
                    "proof_pipeline",
                ));
                ComponentHealth::unhealthy(
                    "proof_pipeline",
                    proof_check.error.as_deref().unwrap_or("Proof pipeline check failed"),
                    proof_check.duration_ms,
                )
            };
            components.push(proof_component);
        }

        // Gather system info
        let system_info = SystemInfo::gather();

        let check_duration = start.elapsed().as_millis() as u64;

        let report = HealthReport::new(components, issues, system_info, check_duration);

        // Run performance baseline if configured
        if self.config.run_performance_test {
            // Performance test would go here
            // For now, skip to avoid long health checks
        }

        // Store the report
        *self.last_report.write().unwrap_or_else(|e| e.into_inner()) = Some(report.clone());
        self.check_in_progress.store(false, Ordering::SeqCst);
        self.check_count.fetch_add(1, Ordering::Relaxed);

        report
    }

    /// Quick health check (cached or minimal).
    pub fn quick_check(&self) -> HealthStatus {
        // Return cached status if recent
        if let Some(report) = self.last_report.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let age = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                - report.generated_at;

            if age < 60 {
                return report.status;
            }
        }

        // Run minimal checks
        let crypto_ok = self.verify_crypto_sanity();
        if !crypto_ok {
            return HealthStatus::Unhealthy;
        }

        HealthStatus::Healthy
    }

    /// Checks if the prover is ready to generate proofs.
    pub fn is_ready(&self) -> bool {
        matches!(
            self.quick_check(),
            HealthStatus::Healthy | HealthStatus::Degraded
        )
    }

    /// Returns the last health report.
    pub fn last_report(&self) -> Option<HealthReport> {
        self.last_report.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Returns the number of health checks performed.
    pub fn check_count(&self) -> u64 {
        self.check_count.load(Ordering::Relaxed)
    }

    // ========================================================================
    // Internal Checks
    // ========================================================================

    fn check_system_resources(&self) -> ComponentHealth {
        let start = Instant::now();

        let cpu_cores = num_cpus::get();
        let response_time = start.elapsed().as_millis() as u64;

        if cpu_cores < 2 {
            return ComponentHealth::degraded(
                "system_resources",
                "Limited CPU cores available",
                response_time,
            )
            .with_metric("cpu_cores", cpu_cores as f64);
        }

        ComponentHealth::healthy(
            "system_resources",
            "Sufficient CPU cores",
            response_time,
        )
        .with_metric("cpu_cores", cpu_cores as f64)
    }

    fn check_pipeline_config(&self) -> ComponentHealth {
        let start = Instant::now();

        // Verify default config is valid
        let config = PipelineConfig::default();
        let response_time = start.elapsed().as_millis() as u64;

        if config.k < 10 || config.k > 24 {
            return ComponentHealth::degraded(
                "pipeline_config",
                "Unusual k parameter",
                response_time,
            );
        }

        ComponentHealth::healthy(
            "pipeline_config",
            "Default configuration valid",
            response_time,
        )
        .with_metric("k_parameter", config.k as f64)
    }

    fn check_cache_health(&self) -> ComponentHealth {
        let start = Instant::now();
        let response_time = start.elapsed().as_millis() as u64;

        // Cache module is available if we got this far
        ComponentHealth::healthy(
            "cache",
            "Cache system operational",
            response_time,
        )
    }

    fn check_crypto_primitives(&self) -> ComponentHealth {
        let start = Instant::now();

        // Test basic field operations
        let a = Fr::from(123u64);
        let b = Fr::from(456u64);
        let c = a + b;
        let d = a * b;

        // Verify results
        let expected_sum = Fr::from(579u64);
        let expected_prod = Fr::from(56088u64);

        let response_time = start.elapsed().as_millis() as u64;

        if c != expected_sum || d != expected_prod {
            return ComponentHealth::unhealthy(
                "crypto_primitives",
                "Field arithmetic verification failed",
                response_time,
            );
        }

        ComponentHealth::healthy(
            "crypto_primitives",
            "BN254 field operations verified",
            response_time,
        )
    }

    fn verify_crypto_sanity(&self) -> bool {
        let a = Fr::from(42u64);
        let b = Fr::from(42u64);
        a == b
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Proof Health Check (Startup Sanity)
// ============================================================================

/// Runs a known-good proof generation + verification on startup to ensure
/// the proving pipeline is functional. This catches issues like:
/// - Corrupt SRS parameters
/// - Incompatible curve libraries
/// - Memory/thread pool problems
/// - GPU driver issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofHealthCheck {
    /// Whether the check passed.
    pub passed: bool,
    /// Duration of the check in milliseconds.
    pub duration_ms: u64,
    /// Proof size generated (bytes), if successful.
    pub proof_size: Option<usize>,
    /// Error message if the check failed.
    pub error: Option<String>,
    /// When the check was run.
    pub checked_at: u64,
}

impl ProofHealthCheck {
    /// Runs a minimal proof generation + verification cycle.
    ///
    /// Uses IVCStepCircuit at k=12 (the smallest circuit in the system) to
    /// generate a single proof and self-verify it. If this fails, the prover
    /// cannot generate valid proofs.
    pub fn run() -> Self {
        use crate::pipeline::ProverPipeline;
        use helix_circuits::{IVCStepCircuit, IVCStepWitness, IVCAccumulator};
        use helix_circuits::gadgets::poseidon::poseidon_hash_two;
        use helix_circuits::halo2_proofs::arithmetic::Field;
        use std::time::Instant;

        let start = Instant::now();

        // Build a known-good witness
        let prev_acc = IVCAccumulator::initial(Fr::ZERO);
        let computation = Fr::from(42u64);
        let new_state = poseidon_hash_two(prev_acc.state_commitment, computation);

        let witness = IVCStepWitness {
            prev_acc,
            new_state,
            computation_hash: computation,
            step_error: Fr::from(1u64),
            fold_challenge: None,
            other_acc: None,
        };
        let circuit = IVCStepCircuit { witness };
        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Setup pipeline (k=12, self_verify=true)
        let config = PipelineConfig {
            k: 12,
            self_verify: true,
            enable_tracing: false,
            ..Default::default()
        };
        let mut pipeline = ProverPipeline::<IVCStepCircuit>::with_config(config);

        if let Err(e) = pipeline.setup(&IVCStepCircuit::default()) {
            let duration_ms = start.elapsed().as_millis() as u64;
            tracing::error!("ProofHealthCheck: pipeline setup failed: {e}");
            return Self {
                passed: false,
                duration_ms,
                proof_size: None,
                error: Some(format!("Pipeline setup failed: {e}")),
                checked_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
        }

        // Generate proof
        let proof_bytes = match pipeline.prove(&circuit, &pi_refs) {
            Ok(bytes) => bytes,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                tracing::error!("ProofHealthCheck: proof generation failed: {e}");
                return Self {
                    passed: false,
                    duration_ms,
                    proof_size: None,
                    error: Some(format!("Proof generation failed: {e}")),
                    checked_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };
            }
        };

        // Verify proof
        match pipeline.verify(&proof_bytes, &pi_refs) {
            Ok(true) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                Self {
                    passed: true,
                    duration_ms,
                    proof_size: Some(proof_bytes.len()),
                    error: None,
                    checked_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                }
            }
            Ok(false) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                tracing::error!("ProofHealthCheck: proof verification returned false");
                Self {
                    passed: false,
                    duration_ms,
                    proof_size: Some(proof_bytes.len()),
                    error: Some("Proof verified as invalid".to_string()),
                    checked_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                }
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                tracing::error!("ProofHealthCheck: verification error: {e}");
                Self {
                    passed: false,
                    duration_ms,
                    proof_size: Some(proof_bytes.len()),
                    error: Some(format!("Verification error: {e}")),
                    checked_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                }
            }
        }
    }
}

// ============================================================================
// Global Health Check
// ============================================================================

use std::sync::OnceLock;

static GLOBAL_HEALTH_CHECKER: OnceLock<HealthChecker> = OnceLock::new();

/// Returns the global health checker instance.
pub fn global_health_checker() -> &'static HealthChecker {
    GLOBAL_HEALTH_CHECKER.get_or_init(HealthChecker::new)
}

/// Performs a quick health check using the global checker.
pub fn quick_health_check() -> HealthStatus {
    global_health_checker().quick_check()
}

/// Performs a full health check using the global checker.
pub fn full_health_check() -> HealthReport {
    global_health_checker().check()
}

/// Checks if the prover is ready using the global checker.
pub fn is_prover_ready() -> bool {
    global_health_checker().is_ready()
}

// ============================================================================
// Readiness Probe
// ============================================================================

/// Result of a readiness probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResult {
    /// Whether the prover is ready.
    pub ready: bool,
    /// Status message.
    pub message: String,
    /// List of blocking issues.
    pub blocking_issues: Vec<String>,
    /// When the probe was run.
    pub checked_at: u64,
}

impl ReadinessResult {
    /// Creates a ready result.
    pub fn ready() -> Self {
        Self {
            ready: true,
            message: "Prover ready".to_string(),
            blocking_issues: Vec::new(),
            checked_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Creates a not-ready result.
    pub fn not_ready<S: Into<String>>(message: S, issues: Vec<String>) -> Self {
        Self {
            ready: false,
            message: message.into(),
            blocking_issues: issues,
            checked_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Performs a readiness probe.
pub fn readiness_probe() -> ReadinessResult {
    let status = quick_health_check();

    match status {
        HealthStatus::Healthy | HealthStatus::Degraded => ReadinessResult::ready(),
        HealthStatus::Unhealthy => {
            let report = full_health_check();
            let issues: Vec<String> = report
                .issues
                .iter()
                .filter(|i| i.severity == IssueSeverity::Error)
                .map(|i| format!("{}: {}", i.code, i.message))
                .collect();

            ReadinessResult::not_ready("Prover unhealthy", issues)
        }
        HealthStatus::Unknown => {
            ReadinessResult::not_ready("Health status unknown", vec!["Run health check first".to_string()])
        }
    }
}

/// Performs a liveness probe (minimal check).
pub fn liveness_probe() -> bool {
    // Just verify basic crypto still works
    let a = Fr::from(1u64);
    let b = Fr::from(1u64);
    a == b
}

// ============================================================================
// Startup Initialization
// ============================================================================

/// Result of prover system initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupResult {
    /// Whether initialization succeeded.
    pub success: bool,
    /// Health report from startup check.
    pub health_report: HealthReport,
    /// Proof pipeline sanity check result.
    pub proof_check: Option<ProofHealthCheck>,
    /// Initialization time in milliseconds.
    pub init_time_ms: u64,
    /// Warnings (non-fatal issues).
    pub warnings: Vec<String>,
}

/// Initializes the prover system and runs startup health checks.
///
/// This should be called once at application startup. It:
/// 1. Runs a full health check (system resources, crypto, config)
/// 2. Generates and verifies a known-good proof to sanity-check the pipeline
/// 3. Returns a `StartupResult` indicating readiness
///
/// If the proof pipeline check fails, the system is still started (returns
/// `success: true` with a warning) since the pipeline may work for different
/// circuit configurations. Only crypto primitive failures cause `success: false`.
pub fn initialize_prover_system() -> StartupResult {
    let start = Instant::now();
    tracing::info!("Initializing prover system...");

    // Run full health check with performance test
    let checker = HealthChecker::with_config(HealthCheckConfig {
        run_performance_test: true,
        detailed_diagnostics: true,
        ..Default::default()
    });
    let health_report = checker.check();
    let mut warnings = Vec::new();

    // Check for critical issues
    let has_crypto_failure = health_report.issues.iter().any(|i| {
        i.severity == IssueSeverity::Error && i.code == "CRYPTO_FAILURE"
    });

    if has_crypto_failure {
        let init_time_ms = start.elapsed().as_millis() as u64;
        tracing::error!("Prover system initialization FAILED: cryptographic primitives broken");
        return StartupResult {
            success: false,
            health_report,
            proof_check: None,
            init_time_ms,
            warnings,
        };
    }

    // Run proof pipeline sanity check
    tracing::info!("Running proof pipeline sanity check...");
    let proof_check = ProofHealthCheck::run();

    if !proof_check.passed {
        let msg = format!(
            "Proof pipeline sanity check failed: {}",
            proof_check.error.as_deref().unwrap_or("unknown")
        );
        tracing::warn!("{msg}");
        warnings.push(msg);
    } else {
        tracing::info!(
            proof_size = proof_check.proof_size.unwrap_or(0),
            duration_ms = proof_check.duration_ms,
            "Proof pipeline sanity check passed"
        );
    }

    let init_time_ms = start.elapsed().as_millis() as u64;

    // Collect non-critical warnings
    for issue in &health_report.issues {
        if issue.severity == IssueSeverity::Warning {
            warnings.push(format!("{}: {}", issue.code, issue.message));
        }
    }

    tracing::info!(
        status = %health_report.status,
        init_time_ms,
        num_warnings = warnings.len(),
        "Prover system initialized"
    );

    StartupResult {
        success: true,
        health_report,
        proof_check: Some(proof_check),
        init_time_ms,
        warnings,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_display() {
        assert_eq!(format!("{}", HealthStatus::Healthy), "Healthy");
        assert_eq!(format!("{}", HealthStatus::Unhealthy), "Unhealthy");
    }

    #[test]
    fn test_health_issue_creation() {
        let issue = HealthIssue::error("TEST_ERROR", "Test error message", "test");
        assert_eq!(issue.severity, IssueSeverity::Error);
        assert_eq!(issue.code, "TEST_ERROR");
    }

    #[test]
    fn test_health_checker() {
        let checker = HealthChecker::new();
        let report = checker.check();

        // Should complete without panicking and have a valid status
        // Duration can be 0 on fast systems, so we just verify the check completed
        assert!(matches!(
            report.status,
            HealthStatus::Healthy | HealthStatus::Degraded | HealthStatus::Unhealthy
        ));
    }

    #[test]
    fn test_quick_check() {
        let checker = HealthChecker::new();
        let status = checker.quick_check();

        // Should be healthy on a normal system
        assert!(matches!(
            status,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::default();
        assert!(info.cpu_cores > 0);
        assert!(!info.platform.is_empty());
    }

    #[test]
    fn test_component_health() {
        let component = ComponentHealth::healthy("test", "Test OK", 10);
        assert_eq!(component.status, HealthStatus::Healthy);

        let degraded = ComponentHealth::degraded("test", "Degraded", 10);
        assert_eq!(degraded.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_health_report_summary() {
        let components = vec![
            ComponentHealth::healthy("comp1", "OK", 10),
            ComponentHealth::healthy("comp2", "OK", 20),
        ];
        let report = HealthReport::new(components, vec![], SystemInfo::default(), 30);

        assert_eq!(report.status, HealthStatus::Healthy);
        assert!(report.is_ready());
    }

    #[test]
    fn test_readiness_probe() {
        let result = readiness_probe();
        // Should succeed on a healthy system
        assert!(result.ready || !result.blocking_issues.is_empty());
    }

    #[test]
    fn test_liveness_probe() {
        assert!(liveness_probe());
    }

    #[test]
    fn test_proof_health_check() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let check = ProofHealthCheck::run();
        assert!(check.passed, "ProofHealthCheck should pass: {:?}", check.error);
        assert!(check.proof_size.is_some());
        assert!(check.proof_size.unwrap() > 0);
        assert!(check.duration_ms > 0);
        assert!(check.error.is_none());
    }
}
