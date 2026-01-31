//! Test harness for multi-crate integration testing.
//!
//! Provides a unified framework for setting up and running integration tests
//! across all HELIX crates with proper resource management and timing.

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use super::fixtures::*;
use super::metrics::*;

/// Test harness configuration.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Enable verbose logging.
    pub verbose: bool,
    /// Maximum test duration before timeout.
    pub timeout: Duration,
    /// Number of parallel workers for stress tests.
    pub parallel_workers: usize,
    /// Capture detailed performance metrics.
    pub capture_metrics: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            timeout: Duration::from_secs(300),
            parallel_workers: 4,
            capture_metrics: true,
            seed: 42,
        }
    }
}

impl HarnessConfig {
    /// Configuration for quick CI tests.
    pub fn ci() -> Self {
        Self {
            verbose: false,
            timeout: Duration::from_secs(60),
            parallel_workers: 2,
            capture_metrics: true,
            seed: 42,
        }
    }

    /// Configuration for local development.
    pub fn dev() -> Self {
        Self {
            verbose: true,
            timeout: Duration::from_secs(600),
            parallel_workers: 8,
            capture_metrics: true,
            seed: 42,
        }
    }

    /// Configuration for stress testing.
    pub fn stress() -> Self {
        Self {
            verbose: false,
            timeout: Duration::from_secs(1800),
            parallel_workers: 16,
            capture_metrics: true,
            seed: 42,
        }
    }
}

/// Result of a test phase.
#[derive(Debug, Clone)]
pub struct PhaseResult {
    pub name: String,
    pub success: bool,
    pub duration: Duration,
    pub message: Option<String>,
    pub metrics: Option<MetricCollection>,
}

impl PhaseResult {
    pub fn success(name: &str, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            success: true,
            duration,
            message: None,
            metrics: None,
        }
    }

    pub fn failure(name: &str, duration: Duration, message: &str) -> Self {
        Self {
            name: name.to_string(),
            success: false,
            duration,
            message: Some(message.to_string()),
            metrics: None,
        }
    }

    pub fn with_metrics(mut self, metrics: MetricCollection) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

/// Comprehensive test result.
#[derive(Debug)]
pub struct TestResult {
    pub test_name: String,
    pub phases: Vec<PhaseResult>,
    pub total_duration: Duration,
    pub success: bool,
    pub summary: String,
}

impl TestResult {
    pub fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            phases: Vec::new(),
            total_duration: Duration::ZERO,
            success: true,
            summary: String::new(),
        }
    }

    pub fn add_phase(&mut self, phase: PhaseResult) {
        if !phase.success {
            self.success = false;
        }
        self.total_duration += phase.duration;
        self.phases.push(phase);
    }

    pub fn finalize(&mut self) {
        let passed = self.phases.iter().filter(|p| p.success).count();
        let total = self.phases.len();
        self.summary = format!(
            "{}: {}/{} phases passed in {:?}",
            self.test_name, passed, total, self.total_duration
        );
    }
}

/// Test harness for integration tests.
pub struct TestHarness {
    config: HarnessConfig,
    start_time: Instant,
    metrics: Arc<RwLock<MetricCollection>>,
    scenario: Option<TestScenario>,
}

impl TestHarness {
    /// Creates a new test harness with default configuration.
    pub fn new() -> Self {
        Self::with_config(HarnessConfig::default())
    }

    /// Creates a new test harness with the given configuration.
    pub fn with_config(config: HarnessConfig) -> Self {
        if config.verbose {
            let subscriber = FmtSubscriber::builder()
                .with_max_level(Level::DEBUG)
                .with_test_writer()
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
        }

        Self {
            config,
            start_time: Instant::now(),
            metrics: Arc::new(RwLock::new(MetricCollection::new())),
            scenario: None,
        }
    }

    /// Sets the test scenario.
    pub fn with_scenario(mut self, scenario: TestScenario) -> Self {
        self.scenario = Some(scenario);
        self
    }

    /// Returns the configuration.
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }

    /// Returns the test scenario.
    pub fn scenario(&self) -> Option<&TestScenario> {
        self.scenario.as_ref()
    }

    /// Returns a clone of the metrics.
    pub fn metrics(&self) -> MetricCollection {
        self.metrics.read().clone()
    }

    /// Records a metric.
    pub fn record_metric(&self, name: &str, value: f64, unit: &str) {
        if self.config.capture_metrics {
            self.metrics.write().record(name, value, unit);
        }
    }

    /// Records a duration metric.
    pub fn record_duration(&self, name: &str, duration: Duration) {
        self.record_metric(name, duration.as_secs_f64() * 1000.0, "ms");
    }

    /// Checks if the test has timed out.
    pub fn check_timeout(&self) -> bool {
        self.start_time.elapsed() > self.config.timeout
    }

    /// Runs a test phase with timing.
    pub fn run_phase<F, T>(&self, name: &str, f: F) -> Result<(T, PhaseResult), PhaseResult>
    where
        F: FnOnce(&Self) -> Result<T, String>,
    {
        let start = Instant::now();

        if self.check_timeout() {
            return Err(PhaseResult::failure(name, Duration::ZERO, "Test timeout"));
        }

        match f(self) {
            Ok(result) => {
                let duration = start.elapsed();
                self.record_duration(&format!("{}_duration", name), duration);
                let phase = PhaseResult::success(name, duration);
                Ok((result, phase))
            }
            Err(msg) => {
                let duration = start.elapsed();
                Err(PhaseResult::failure(name, duration, &msg))
            }
        }
    }

    /// Runs an async test phase with timing.
    pub async fn run_phase_async<F, Fut, T>(
        &self,
        name: &str,
        f: F,
    ) -> Result<(T, PhaseResult), PhaseResult>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, String>>,
    {
        let start = Instant::now();

        if self.check_timeout() {
            return Err(PhaseResult::failure(name, Duration::ZERO, "Test timeout"));
        }

        match f().await {
            Ok(result) => {
                let duration = start.elapsed();
                self.record_duration(&format!("{}_duration", name), duration);
                let phase = PhaseResult::success(name, duration);
                Ok((result, phase))
            }
            Err(msg) => {
                let duration = start.elapsed();
                Err(PhaseResult::failure(name, duration, &msg))
            }
        }
    }

    /// Times a block of code.
    pub fn timed<F, T>(&self, name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        self.record_duration(name, start.elapsed());
        result
    }

    /// Generates a comprehensive test report.
    pub fn generate_report(&self, result: &TestResult) -> String {
        let mut report = String::new();

        report.push_str(&format!("\n{'='*60}\n"));
        report.push_str(&format!("Test Report: {}\n", result.test_name));
        report.push_str(&format!("{'='*60}\n\n"));

        report.push_str(&format!("Status: {}\n", if result.success { "PASSED" } else { "FAILED" }));
        report.push_str(&format!("Total Duration: {:?}\n\n", result.total_duration));

        report.push_str("Phases:\n");
        for phase in &result.phases {
            let status = if phase.success { "✓" } else { "✗" };
            report.push_str(&format!(
                "  {} {} ({:?})\n",
                status, phase.name, phase.duration
            ));
            if let Some(ref msg) = phase.message {
                report.push_str(&format!("      Error: {}\n", msg));
            }
        }

        if self.config.capture_metrics {
            report.push_str("\nMetrics:\n");
            let metrics = self.metrics.read();
            for metric in metrics.all() {
                report.push_str(&format!(
                    "  {}: {:.3} {}\n",
                    metric.name, metric.value, metric.unit
                ));
            }
        }

        report.push_str(&format!("\n{'='*60}\n"));
        report.push_str(&format!("Summary: {}\n", result.summary));

        report
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Test runner for executing multiple test scenarios.
pub struct TestRunner {
    harness: TestHarness,
    results: Vec<TestResult>,
}

impl TestRunner {
    pub fn new(config: HarnessConfig) -> Self {
        Self {
            harness: TestHarness::with_config(config),
            results: Vec::new(),
        }
    }

    pub fn harness(&self) -> &TestHarness {
        &self.harness
    }

    pub fn add_result(&mut self, result: TestResult) {
        self.results.push(result);
    }

    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.success)
    }

    pub fn generate_summary(&self) -> String {
        let passed = self.results.iter().filter(|r| r.success).count();
        let total = self.results.len();
        let mut summary = format!("\n{'='*60}\n");
        summary.push_str(&format!("TEST SUITE SUMMARY: {}/{} tests passed\n", passed, total));
        summary.push_str(&format!("{'='*60}\n\n"));

        for result in &self.results {
            let status = if result.success { "PASS" } else { "FAIL" };
            summary.push_str(&format!(
                "[{}] {} ({:?})\n",
                status, result.test_name, result.total_duration
            ));
        }

        summary
    }
}

/// Macro to run a test phase and handle errors.
#[macro_export]
macro_rules! run_phase {
    ($harness:expr, $result:expr, $name:expr, $body:block) => {
        match $harness.run_phase($name, |_| {
            let phase_result: Result<(), String> = (|| $body)();
            phase_result
        }) {
            Ok((_, phase)) => $result.add_phase(phase),
            Err(phase) => {
                $result.add_phase(phase);
                return $result;
            }
        }
    };
}

/// Macro to assert with test result error message.
#[macro_export]
macro_rules! test_assert {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            return Err($msg.to_string());
        }
    };
    ($cond:expr) => {
        test_assert!($cond, stringify!($cond))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_creation() {
        let harness = TestHarness::new();
        assert!(!harness.check_timeout());
    }

    #[test]
    fn test_phase_result() {
        let phase = PhaseResult::success("test_phase", Duration::from_millis(100));
        assert!(phase.success);
        assert_eq!(phase.name, "test_phase");
    }

    #[test]
    fn test_metric_recording() {
        let harness = TestHarness::new();
        harness.record_metric("test_metric", 42.0, "ms");

        let metrics = harness.metrics();
        assert_eq!(metrics.all().len(), 1);
    }

    #[test]
    fn test_timed_block() {
        let harness = TestHarness::new();
        let result = harness.timed("computation", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        assert_eq!(result, 42);
    }
}
