//! Benchmark Harness with Standardized Reporting
//!
//! Provides a unified benchmark harness that generates consistent reports in
//! multiple formats (JSON, Markdown, Terminal) with regression detection.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::{
    MAX_REGRESSION_PERCENT, REGRESSION_ALERT_THRESHOLD, TARGET_OVERHEAD_MULTIPLE,
    TARGET_PROOF_TIME_MS,
};

/// Configuration for the benchmark harness.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Name of this benchmark suite.
    pub suite_name: String,
    /// Number of warm-up iterations.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measure_iterations: usize,
    /// Output directory for reports.
    pub output_dir: PathBuf,
    /// Whether to save baseline on completion.
    pub save_baseline: bool,
    /// Path to baseline file for comparison.
    pub baseline_path: Option<PathBuf>,
    /// Output formats to generate.
    pub output_formats: Vec<OutputFormat>,
    /// Whether to fail on regression.
    pub fail_on_regression: bool,
    /// Regression threshold percentage.
    pub regression_threshold_pct: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            suite_name: "helix_benchmarks".to_string(),
            warmup_iterations: 3,
            measure_iterations: 10,
            output_dir: PathBuf::from("target/helix-benchmarks"),
            save_baseline: false,
            baseline_path: None,
            output_formats: vec![OutputFormat::Terminal, OutputFormat::Json],
            fail_on_regression: false,
            regression_threshold_pct: MAX_REGRESSION_PERCENT,
        }
    }
}

impl BenchmarkConfig {
    /// Creates a config for CI runs.
    pub fn ci() -> Self {
        Self {
            suite_name: "helix_ci_benchmarks".to_string(),
            warmup_iterations: 2,
            measure_iterations: 5,
            output_dir: PathBuf::from("target/helix-benchmarks"),
            save_baseline: true,
            baseline_path: Some(PathBuf::from("target/helix-benchmarks/baseline.json")),
            output_formats: vec![OutputFormat::Json, OutputFormat::Markdown],
            fail_on_regression: true,
            regression_threshold_pct: MAX_REGRESSION_PERCENT,
        }
    }

    /// Creates a config for development (faster).
    pub fn dev() -> Self {
        Self {
            suite_name: "helix_dev_benchmarks".to_string(),
            warmup_iterations: 1,
            measure_iterations: 3,
            output_dir: PathBuf::from("target/helix-benchmarks"),
            save_baseline: false,
            baseline_path: None,
            output_formats: vec![OutputFormat::Terminal],
            fail_on_regression: false,
            regression_threshold_pct: 20.0,
        }
    }

    /// Creates a config for precise measurements.
    pub fn precise() -> Self {
        Self {
            suite_name: "helix_precise_benchmarks".to_string(),
            warmup_iterations: 5,
            measure_iterations: 50,
            output_dir: PathBuf::from("target/helix-benchmarks"),
            save_baseline: true,
            baseline_path: None,
            output_formats: vec![
                OutputFormat::Json,
                OutputFormat::Markdown,
                OutputFormat::Terminal,
            ],
            fail_on_regression: false,
            regression_threshold_pct: MAX_REGRESSION_PERCENT,
        }
    }
}

/// Output format for benchmark reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Terminal,
    Json,
    Markdown,
    Csv,
}

/// Status of regression detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionStatus {
    /// No baseline available for comparison.
    NoBaseline,
    /// Performance is stable (within threshold).
    Stable,
    /// Performance improved.
    Improved,
    /// Performance regressed.
    Regressed,
}

/// Metrics collected for a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Mean execution time in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds.
    pub stddev_ns: f64,
    /// Minimum time in nanoseconds.
    pub min_ns: f64,
    /// Maximum time in nanoseconds.
    pub max_ns: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Throughput (operations per second) if applicable.
    pub throughput_ops: Option<f64>,
    /// Memory usage in bytes if measured.
    pub memory_bytes: Option<usize>,
    /// Custom tags for categorization.
    pub tags: BTreeMap<String, String>,
}

impl BenchmarkMetrics {
    /// Creates metrics from a series of duration samples.
    pub fn from_durations(samples: &[Duration]) -> Self {
        let n = samples.len() as f64;
        let nanos: Vec<f64> = samples.iter().map(|d| d.as_nanos() as f64).collect();

        let mean_ns = nanos.iter().sum::<f64>() / n;
        let variance = nanos.iter().map(|x| (x - mean_ns).powi(2)).sum::<f64>() / n;
        let stddev_ns = variance.sqrt();

        let min_ns = nanos.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ns = nanos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            mean_ns,
            stddev_ns,
            min_ns,
            max_ns,
            iterations: samples.len(),
            throughput_ops: None,
            memory_bytes: None,
            tags: BTreeMap::new(),
        }
    }

    /// Returns mean time as Duration.
    pub fn mean_duration(&self) -> Duration {
        Duration::from_nanos(self.mean_ns as u64)
    }

    /// Returns mean time in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean_ns / 1_000_000.0
    }

    /// Returns mean time in microseconds.
    pub fn mean_us(&self) -> f64 {
        self.mean_ns / 1_000.0
    }

    /// Adds a tag to the metrics.
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Sets throughput.
    pub fn with_throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput_ops = Some(ops_per_sec);
        self
    }

    /// Sets memory usage.
    pub fn with_memory(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }
}

/// Result of a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Category (e.g., "gkr", "halo2", "native").
    pub category: String,
    /// Collected metrics.
    pub metrics: BenchmarkMetrics,
    /// Regression status compared to baseline.
    pub regression_status: RegressionStatus,
    /// Percentage change from baseline (if available).
    pub change_pct: Option<f64>,
    /// Timestamp when benchmark was run.
    pub timestamp: String,
}

impl BenchmarkResult {
    pub fn new(name: &str, category: &str, metrics: BenchmarkMetrics) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            metrics,
            regression_status: RegressionStatus::NoBaseline,
            change_pct: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Compares this result to a baseline and sets regression status.
    pub fn compare_to_baseline(&mut self, baseline: &BenchmarkMetrics, threshold_pct: f64) {
        let change = (self.metrics.mean_ns - baseline.mean_ns) / baseline.mean_ns * 100.0;
        self.change_pct = Some(change);

        self.regression_status = if change > threshold_pct {
            RegressionStatus::Regressed
        } else if change < -threshold_pct {
            RegressionStatus::Improved
        } else {
            RegressionStatus::Stable
        };
    }
}

/// Complete benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Suite name.
    pub suite_name: String,
    /// Version of the report format.
    pub version: u32,
    /// Git commit hash if available.
    pub git_commit: Option<String>,
    /// Timestamp of the report.
    pub timestamp: String,
    /// All benchmark results.
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics.
    pub summary: ReportSummary,
}

/// Summary statistics for a benchmark report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total benchmarks run.
    pub total_benchmarks: usize,
    /// Number of regressions detected.
    pub regressions: usize,
    /// Number of improvements detected.
    pub improvements: usize,
    /// Number of stable results.
    pub stable: usize,
    /// Number without baseline.
    pub no_baseline: usize,
    /// Average overhead multiple (if applicable).
    pub avg_overhead: Option<f64>,
    /// Maximum overhead multiple.
    pub max_overhead: Option<f64>,
    /// Whether all benchmarks passed (no regressions).
    pub passed: bool,
}

impl BenchmarkReport {
    pub fn new(suite_name: &str) -> Self {
        Self {
            suite_name: suite_name.to_string(),
            version: 1,
            git_commit: std::env::var("GITHUB_SHA").ok().or_else(|| {
                // Try to get from git command
                std::process::Command::new("git")
                    .args(["rev-parse", "HEAD"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
            }),
            timestamp: chrono::Utc::now().to_rfc3339(),
            results: Vec::new(),
            summary: ReportSummary::default(),
        }
    }

    /// Adds a result to the report.
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Finalizes the report by computing summary statistics.
    pub fn finalize(&mut self) {
        let mut regressions = 0;
        let mut improvements = 0;
        let mut stable = 0;
        let mut no_baseline = 0;

        for result in &self.results {
            match result.regression_status {
                RegressionStatus::Regressed => regressions += 1,
                RegressionStatus::Improved => improvements += 1,
                RegressionStatus::Stable => stable += 1,
                RegressionStatus::NoBaseline => no_baseline += 1,
            }
        }

        self.summary = ReportSummary {
            total_benchmarks: self.results.len(),
            regressions,
            improvements,
            stable,
            no_baseline,
            avg_overhead: None,
            max_overhead: None,
            passed: regressions == 0,
        };
    }

    /// Saves the report to a JSON file.
    pub fn save_json(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads a report from a JSON file.
    pub fn load_json(path: &PathBuf) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let report: Self = serde_json::from_reader(reader)?;
        Ok(report)
    }

    /// Generates a markdown report.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!(
            "# HELIX Benchmark Report: {}\n\n",
            self.suite_name
        ));
        md.push_str(&format!("**Generated:** {}\n\n", self.timestamp));

        if let Some(ref commit) = self.git_commit {
            md.push_str(&format!("**Commit:** `{}`\n\n", commit));
        }

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("| Metric | Value |\n"));
        md.push_str(&format!("|--------|-------|\n"));
        md.push_str(&format!(
            "| Total Benchmarks | {} |\n",
            self.summary.total_benchmarks
        ));
        md.push_str(&format!(
            "| Regressions | {} {} |\n",
            self.summary.regressions,
            if self.summary.regressions > 0 {
                "🔴"
            } else {
                ""
            }
        ));
        md.push_str(&format!(
            "| Improvements | {} {} |\n",
            self.summary.improvements,
            if self.summary.improvements > 0 {
                "🟢"
            } else {
                ""
            }
        ));
        md.push_str(&format!("| Stable | {} |\n", self.summary.stable));
        md.push_str(&format!(
            "| Status | {} |\n\n",
            if self.summary.passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));

        // Results table
        md.push_str("## Results\n\n");
        md.push_str("| Benchmark | Category | Mean (ms) | Stddev | Change | Status |\n");
        md.push_str("|-----------|----------|-----------|--------|--------|--------|\n");

        for result in &self.results {
            let status_icon = match result.regression_status {
                RegressionStatus::Regressed => "🔴",
                RegressionStatus::Improved => "🟢",
                RegressionStatus::Stable => "⚪",
                RegressionStatus::NoBaseline => "➖",
            };

            let change_str = result
                .change_pct
                .map(|c| format!("{:+.1}%", c))
                .unwrap_or_else(|| "N/A".to_string());

            md.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} | {} | {} |\n",
                result.name,
                result.category,
                result.metrics.mean_ms(),
                result.metrics.stddev_ns / 1_000_000.0,
                change_str,
                status_icon,
            ));
        }

        md.push_str("\n## Overhead Analysis\n\n");
        md.push_str(&format!(
            "- **Target overhead:** ≤{:.0}x\n",
            TARGET_OVERHEAD_MULTIPLE
        ));
        md.push_str(&format!(
            "- **Alert threshold:** ≤{:.0}x\n",
            REGRESSION_ALERT_THRESHOLD
        ));
        md.push_str(&format!(
            "- **Target proof time:** ≤{}ms per step\n\n",
            TARGET_PROOF_TIME_MS
        ));

        md
    }

    /// Prints a terminal-friendly report.
    pub fn print_terminal(&self) {
        println!("\n{}", "=".repeat(70));
        println!("HELIX Benchmark Report: {}", self.suite_name);
        println!("{}", "=".repeat(70));
        println!("Generated: {}", self.timestamp);
        if let Some(ref commit) = self.git_commit {
            println!("Commit: {}", commit);
        }
        println!();

        // Summary box
        println!("┌─────────────────────────────────────────────┐");
        println!("│ Summary                                     │");
        println!("├─────────────────────────────────────────────┤");
        println!(
            "│ Total:       {:>5}                          │",
            self.summary.total_benchmarks
        );
        println!(
            "│ Regressions: {:>5} {}                          │",
            self.summary.regressions,
            if self.summary.regressions > 0 {
                "(!)"
            } else {
                "   "
            }
        );
        println!(
            "│ Improvements:{:>5}                          │",
            self.summary.improvements
        );
        println!(
            "│ Stable:      {:>5}                          │",
            self.summary.stable
        );
        println!(
            "│ Status:      {}                       │",
            if self.summary.passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
        println!("└─────────────────────────────────────────────┘");
        println!();

        // Results
        println!(
            "{:<40} {:>12} {:>12} {:>10}",
            "Benchmark", "Mean (ms)", "Stddev", "Change"
        );
        println!("{}", "-".repeat(78));

        for result in &self.results {
            let status = match result.regression_status {
                RegressionStatus::Regressed => "[!]",
                RegressionStatus::Improved => "[+]",
                RegressionStatus::Stable => "[ ]",
                RegressionStatus::NoBaseline => "[-]",
            };

            let change_str = result
                .change_pct
                .map(|c| format!("{:+.1}%", c))
                .unwrap_or_else(|| "N/A".to_string());

            println!(
                "{} {:<37} {:>12.3} {:>12.3} {:>10}",
                status,
                result.name,
                result.metrics.mean_ms(),
                result.metrics.stddev_ns / 1_000_000.0,
                change_str,
            );
        }

        println!("{}", "=".repeat(70));
    }
}

/// The main benchmark harness.
pub struct BenchmarkHarness {
    config: BenchmarkConfig,
    report: BenchmarkReport,
    baseline: Option<BenchmarkReport>,
    start_time: Instant,
}

impl BenchmarkHarness {
    /// Creates a new benchmark harness with the given configuration.
    pub fn new(config: BenchmarkConfig) -> Self {
        let report = BenchmarkReport::new(&config.suite_name);

        // Load baseline if specified
        let baseline = config
            .baseline_path
            .as_ref()
            .and_then(|path| BenchmarkReport::load_json(path).ok());

        Self {
            config,
            report,
            baseline,
            start_time: Instant::now(),
        }
    }

    /// Creates a harness with default configuration.
    pub fn default_config() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Runs a benchmark function and records the result.
    pub fn run_benchmark<F, R>(&mut self, name: &str, category: &str, mut f: F) -> &BenchmarkResult
    where
        F: FnMut() -> R,
    {
        // Warm-up
        for _ in 0..self.config.warmup_iterations {
            let _ = std::hint::black_box(f());
        }

        // Measure
        let mut times = Vec::with_capacity(self.config.measure_iterations);
        for _ in 0..self.config.measure_iterations {
            let start = Instant::now();
            let _ = std::hint::black_box(f());
            times.push(start.elapsed());
        }

        let metrics = BenchmarkMetrics::from_durations(&times);
        let mut result = BenchmarkResult::new(name, category, metrics);

        // Compare to baseline if available
        if let Some(ref baseline) = self.baseline {
            if let Some(baseline_result) = baseline.results.iter().find(|r| r.name == name) {
                result.compare_to_baseline(
                    &baseline_result.metrics,
                    self.config.regression_threshold_pct,
                );
            }
        }

        self.report.add_result(result);
        self.report.results.last().unwrap()
    }

    /// Runs a benchmark with custom metrics (for cases where timing is done externally).
    pub fn record_benchmark(
        &mut self,
        name: &str,
        category: &str,
        metrics: BenchmarkMetrics,
    ) -> &BenchmarkResult {
        let mut result = BenchmarkResult::new(name, category, metrics);

        if let Some(ref baseline) = self.baseline {
            if let Some(baseline_result) = baseline.results.iter().find(|r| r.name == name) {
                result.compare_to_baseline(
                    &baseline_result.metrics,
                    self.config.regression_threshold_pct,
                );
            }
        }

        self.report.add_result(result);
        self.report.results.last().unwrap()
    }

    /// Finalizes the benchmark run and generates reports.
    pub fn finalize(&mut self) -> BenchmarkOutcome {
        self.report.finalize();

        // Create output directory
        let _ = fs::create_dir_all(&self.config.output_dir);

        // Generate outputs in requested formats
        for format in &self.config.output_formats {
            match format {
                OutputFormat::Terminal => self.report.print_terminal(),
                OutputFormat::Json => {
                    let path = self.config.output_dir.join("report.json");
                    if let Err(e) = self.report.save_json(&path) {
                        eprintln!("Failed to save JSON report: {}", e);
                    }
                }
                OutputFormat::Markdown => {
                    let md = self.report.to_markdown();
                    let path = self.config.output_dir.join("report.md");
                    if let Err(e) = fs::write(&path, md) {
                        eprintln!("Failed to save markdown report: {}", e);
                    }
                }
                OutputFormat::Csv => {
                    let path = self.config.output_dir.join("report.csv");
                    if let Err(e) = self.save_csv(&path) {
                        eprintln!("Failed to save CSV report: {}", e);
                    }
                }
            }
        }

        // Save baseline if configured
        if self.config.save_baseline {
            let baseline_path = self.config.output_dir.join("baseline.json");
            if let Err(e) = self.report.save_json(&baseline_path) {
                eprintln!("Failed to save baseline: {}", e);
            }
        }

        let duration = self.start_time.elapsed();

        BenchmarkOutcome {
            passed: self.report.summary.passed || !self.config.fail_on_regression,
            total_duration: duration,
            regressions: self.report.summary.regressions,
            improvements: self.report.summary.improvements,
        }
    }

    /// Saves the report as CSV.
    fn save_csv(&self, path: &PathBuf) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "name,category,mean_ns,stddev_ns,min_ns,max_ns,iterations,change_pct,status"
        )?;

        for result in &self.report.results {
            let status = match result.regression_status {
                RegressionStatus::Regressed => "regressed",
                RegressionStatus::Improved => "improved",
                RegressionStatus::Stable => "stable",
                RegressionStatus::NoBaseline => "no_baseline",
            };
            let change = result.change_pct.map(|c| c.to_string()).unwrap_or_default();

            writeln!(
                file,
                "{},{},{:.2},{:.2},{:.2},{:.2},{},{},{}",
                result.name,
                result.category,
                result.metrics.mean_ns,
                result.metrics.stddev_ns,
                result.metrics.min_ns,
                result.metrics.max_ns,
                result.metrics.iterations,
                change,
                status,
            )?;
        }

        Ok(())
    }

    /// Returns the current report.
    pub fn report(&self) -> &BenchmarkReport {
        &self.report
    }
}

/// Outcome of a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkOutcome {
    /// Whether the benchmark passed (no regressions or regression checking disabled).
    pub passed: bool,
    /// Total duration of the benchmark run.
    pub total_duration: Duration,
    /// Number of regressions detected.
    pub regressions: usize,
    /// Number of improvements detected.
    pub improvements: usize,
}

impl BenchmarkOutcome {
    /// Asserts that the benchmark passed, panicking if it didn't.
    pub fn assert_passed(&self) {
        if !self.passed {
            panic!(
                "Benchmark failed with {} regressions (threshold: {}%)",
                self.regressions, MAX_REGRESSION_PERCENT
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_metrics_from_durations() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(9),
            Duration::from_millis(10),
        ];

        let metrics = BenchmarkMetrics::from_durations(&samples);

        assert!((metrics.mean_ms() - 10.4).abs() < 0.1);
        assert_eq!(metrics.iterations, 5);
    }

    #[test]
    fn test_benchmark_result_regression() {
        let mut result = BenchmarkResult::new(
            "test",
            "unit",
            BenchmarkMetrics {
                mean_ns: 1_100_000.0,
                stddev_ns: 10_000.0,
                min_ns: 1_000_000.0,
                max_ns: 1_200_000.0,
                iterations: 10,
                throughput_ops: None,
                memory_bytes: None,
                tags: BTreeMap::new(),
            },
        );

        let baseline = BenchmarkMetrics {
            mean_ns: 1_000_000.0,
            stddev_ns: 10_000.0,
            min_ns: 900_000.0,
            max_ns: 1_100_000.0,
            iterations: 10,
            throughput_ops: None,
            memory_bytes: None,
            tags: BTreeMap::new(),
        };

        // 10% regression threshold
        result.compare_to_baseline(&baseline, 10.0);
        assert_eq!(result.regression_status, RegressionStatus::Stable);

        // 5% regression threshold - should trigger regression
        result.compare_to_baseline(&baseline, 5.0);
        assert_eq!(result.regression_status, RegressionStatus::Regressed);
    }

    #[test]
    fn test_harness_run_benchmark() {
        let mut harness = BenchmarkHarness::new(BenchmarkConfig {
            warmup_iterations: 1,
            measure_iterations: 3,
            ..BenchmarkConfig::default()
        });

        let result = harness.run_benchmark("test_bench", "unit", || {
            std::thread::sleep(Duration::from_micros(100));
        });

        assert!(result.metrics.mean_us() >= 50.0);
    }

    #[test]
    fn test_report_markdown_generation() {
        let mut report = BenchmarkReport::new("test_suite");
        report.add_result(BenchmarkResult::new(
            "test_1",
            "unit",
            BenchmarkMetrics {
                mean_ns: 1_000_000.0,
                stddev_ns: 10_000.0,
                min_ns: 900_000.0,
                max_ns: 1_100_000.0,
                iterations: 10,
                throughput_ops: None,
                memory_bytes: None,
                tags: BTreeMap::new(),
            },
        ));
        report.finalize();

        let md = report.to_markdown();
        assert!(md.contains("HELIX Benchmark Report"));
        assert!(md.contains("test_1"));
    }
}
