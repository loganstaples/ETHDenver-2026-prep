//! Performance metrics collection and analysis for integration tests.
//!
//! Provides utilities for tracking timing, memory usage, proof sizes, and
//! comparing against baseline performance.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// A single metric measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub timestamp_ms: u64,
}

impl Metric {
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
            timestamp_ms,
        }
    }
}

/// Collection of metrics with analysis capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricCollection {
    metrics: Vec<Metric>,
    /// Baseline values for regression detection.
    baselines: HashMap<String, f64>,
}

impl MetricCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a collection with baseline values.
    pub fn with_baselines(baselines: HashMap<String, f64>) -> Self {
        Self {
            metrics: Vec::new(),
            baselines,
        }
    }

    /// Records a metric.
    pub fn record(&mut self, name: &str, value: f64, unit: &str) {
        self.metrics.push(Metric::new(name, value, unit));
    }

    /// Records a duration in milliseconds.
    pub fn record_duration(&mut self, name: &str, duration: Duration) {
        self.record(name, duration.as_secs_f64() * 1000.0, "ms");
    }

    /// Records a memory size in bytes.
    pub fn record_bytes(&mut self, name: &str, bytes: usize) {
        self.record(name, bytes as f64, "bytes");
    }

    /// Returns all metrics.
    pub fn all(&self) -> &[Metric] {
        &self.metrics
    }

    /// Returns metrics by name.
    pub fn get(&self, name: &str) -> Option<&Metric> {
        self.metrics.iter().rev().find(|m| m.name == name)
    }

    /// Returns all metrics with a given prefix.
    pub fn with_prefix(&self, prefix: &str) -> Vec<&Metric> {
        self.metrics
            .iter()
            .filter(|m| m.name.starts_with(prefix))
            .collect()
    }

    /// Computes statistics for a metric series.
    pub fn stats(&self, name: &str) -> Option<MetricStats> {
        let values: Vec<f64> = self.metrics
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.value)
            .collect();

        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let stddev = variance.sqrt();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
        let p99_idx = ((count as f64 * 0.99) as usize).min(count - 1);

        Some(MetricStats {
            count,
            min,
            max,
            mean,
            median,
            stddev,
            p95: sorted[p95_idx],
            p99: sorted[p99_idx],
        })
    }

    /// Checks for performance regression against baselines.
    pub fn check_regression(&self, threshold_pct: f64) -> Vec<RegressionReport> {
        let mut regressions = Vec::new();

        for (name, baseline) in &self.baselines {
            if let Some(metric) = self.get(name) {
                let pct_change = (metric.value - baseline) / baseline * 100.0;
                if pct_change > threshold_pct {
                    regressions.push(RegressionReport {
                        metric_name: name.clone(),
                        baseline: *baseline,
                        current: metric.value,
                        change_pct: pct_change,
                        is_regression: true,
                    });
                }
            }
        }

        regressions
    }

    /// Merges another collection into this one.
    pub fn merge(&mut self, other: &MetricCollection) {
        self.metrics.extend(other.metrics.clone());
    }

    /// Exports to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Imports from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Statistics for a metric series.
#[derive(Debug, Clone)]
pub struct MetricStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub p95: f64,
    pub p99: f64,
}

impl std::fmt::Display for MetricStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "count={}, min={:.2}, max={:.2}, mean={:.2}, median={:.2}, p95={:.2}, p99={:.2}",
            self.count, self.min, self.max, self.mean, self.median, self.p95, self.p99
        )
    }
}

/// Performance regression report.
#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub metric_name: String,
    pub baseline: f64,
    pub current: f64,
    pub change_pct: f64,
    pub is_regression: bool,
}

impl std::fmt::Display for RegressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = if self.is_regression { "⚠️" } else { "✓" };
        write!(
            f,
            "{} {}: {:.2} -> {:.2} ({:+.1}%)",
            symbol, self.metric_name, self.baseline, self.current, self.change_pct
        )
    }
}

/// Timer for measuring code blocks.
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    /// Starts a new timer.
    pub fn start(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Stops the timer and returns the duration.
    pub fn stop(self) -> (String, Duration) {
        (self.name, self.start.elapsed())
    }

    /// Stops the timer and records to a collection.
    pub fn stop_and_record(self, collection: &mut MetricCollection) {
        let duration = self.start.elapsed();
        collection.record_duration(&self.name, duration);
    }
}

/// Standard performance baselines for HELIX.
pub struct PerformanceBaselines;

impl PerformanceBaselines {
    /// Returns baselines for proof generation.
    pub fn proof_generation() -> HashMap<String, f64> {
        let mut baselines = HashMap::new();
        baselines.insert("proof_generation_tiny_ms".to_string(), 500.0);
        baselines.insert("proof_generation_small_ms".to_string(), 2000.0);
        baselines.insert("proof_generation_medium_ms".to_string(), 10000.0);
        baselines
    }

    /// Returns baselines for proof verification.
    pub fn proof_verification() -> HashMap<String, f64> {
        let mut baselines = HashMap::new();
        baselines.insert("native_verification_ms".to_string(), 100.0);
        baselines.insert("evm_verification_gas".to_string(), 300000.0);
        baselines
    }

    /// Returns baselines for MPC operations.
    pub fn mpc_operations() -> HashMap<String, f64> {
        let mut baselines = HashMap::new();
        baselines.insert("secret_sharing_ms".to_string(), 10.0);
        baselines.insert("share_reconstruction_ms".to_string(), 10.0);
        baselines.insert("secure_matmul_ms".to_string(), 100.0);
        baselines
    }

    /// Returns baselines for training.
    pub fn training() -> HashMap<String, f64> {
        let mut baselines = HashMap::new();
        baselines.insert("training_step_with_proof_ms".to_string(), 1000.0);
        baselines.insert("gradient_aggregation_ms".to_string(), 50.0);
        baselines.insert("checkpoint_save_ms".to_string(), 100.0);
        baselines.insert("checkpoint_load_ms".to_string(), 100.0);
        baselines
    }

    /// Returns all baselines combined.
    pub fn all() -> HashMap<String, f64> {
        let mut all = HashMap::new();
        all.extend(Self::proof_generation());
        all.extend(Self::proof_verification());
        all.extend(Self::mpc_operations());
        all.extend(Self::training());
        all
    }
}

/// Overhead calculator for comparing ZK vs native execution.
#[derive(Debug)]
pub struct OverheadCalculator {
    native_time: Option<Duration>,
    zk_time: Option<Duration>,
}

impl OverheadCalculator {
    pub fn new() -> Self {
        Self {
            native_time: None,
            zk_time: None,
        }
    }

    pub fn set_native_time(&mut self, duration: Duration) {
        self.native_time = Some(duration);
    }

    pub fn set_zk_time(&mut self, duration: Duration) {
        self.zk_time = Some(duration);
    }

    /// Calculates the overhead factor (ZK time / native time).
    pub fn overhead_factor(&self) -> Option<f64> {
        match (self.native_time, self.zk_time) {
            (Some(native), Some(zk)) if !native.is_zero() => {
                Some(zk.as_secs_f64() / native.as_secs_f64())
            }
            _ => None,
        }
    }

    /// Returns true if overhead is within the target (30x for HELIX).
    pub fn within_target(&self, target: f64) -> bool {
        self.overhead_factor().map(|f| f <= target).unwrap_or(false)
    }
}

impl Default for OverheadCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_collection() {
        let mut collection = MetricCollection::new();
        collection.record("test_metric", 42.0, "ms");
        collection.record("test_metric", 43.0, "ms");

        assert_eq!(collection.all().len(), 2);
        assert_eq!(collection.get("test_metric").unwrap().value, 43.0);
    }

    #[test]
    fn test_metric_stats() {
        let mut collection = MetricCollection::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            collection.record("values", v, "unit");
        }

        let stats = collection.stats("values").unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
    }

    #[test]
    fn test_regression_detection() {
        let mut baselines = HashMap::new();
        baselines.insert("metric".to_string(), 100.0);

        let mut collection = MetricCollection::with_baselines(baselines);
        collection.record("metric", 150.0, "ms"); // 50% increase

        let regressions = collection.check_regression(10.0);
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].is_regression);
    }

    #[test]
    fn test_timer() {
        let mut collection = MetricCollection::new();
        let timer = Timer::start("test_op");
        std::thread::sleep(Duration::from_millis(10));
        timer.stop_and_record(&mut collection);

        let metric = collection.get("test_op").unwrap();
        assert!(metric.value >= 10.0);
    }

    #[test]
    fn test_overhead_calculator() {
        let mut calc = OverheadCalculator::new();
        calc.set_native_time(Duration::from_millis(100));
        calc.set_zk_time(Duration::from_millis(2000));

        let factor = calc.overhead_factor().unwrap();
        assert!((factor - 20.0).abs() < 0.1);
        assert!(calc.within_target(30.0));
    }

    #[test]
    fn test_json_serialization() {
        let mut collection = MetricCollection::new();
        collection.record("test", 42.0, "ms");

        let json = collection.to_json().unwrap();
        let recovered = MetricCollection::from_json(&json).unwrap();

        assert_eq!(recovered.all().len(), 1);
    }
}
