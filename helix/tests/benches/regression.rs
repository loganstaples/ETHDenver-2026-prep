//! Regression Detection and CI Integration
//!
//! This module provides automated regression detection by comparing current
//! benchmark results against historical baselines. It integrates with CI/CD
//! pipelines to:
//!
//! - Detect performance regressions before merge
//! - Track overhead trends over time
//! - Alert when overhead exceeds target (30x) or alert threshold (35x)
//! - Generate detailed regression reports
//!
//! # Exit Codes (for CI)
//!
//! - 0: All benchmarks passed, no regressions
//! - 1: Target threshold exceeded (>30x overhead)
//! - 2: Alert threshold exceeded (>35x overhead)
//! - 3: Regression detected vs baseline

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{
    BenchmarkReport, BenchmarkResult, REGRESSION_ALERT_THRESHOLD, TARGET_OVERHEAD_MULTIPLE,
    MAX_REGRESSION_PERCENT,
};

/// Regression detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub benchmark_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub change_percent: f64,
    pub regression_detected: bool,
    pub severity: RegressionSeverity,
}

impl RegressionResult {
    pub fn new(name: &str, current: f64, baseline: f64, threshold: f64) -> Self {
        let change = if baseline > 0.0 {
            (current - baseline) / baseline * 100.0
        } else {
            0.0
        };

        let regression = change > threshold;
        let severity = if change > threshold * 2.0 {
            RegressionSeverity::Critical
        } else if change > threshold {
            RegressionSeverity::Warning
        } else if change < -threshold {
            RegressionSeverity::Improvement
        } else {
            RegressionSeverity::Stable
        };

        Self {
            benchmark_name: name.to_string(),
            current_value: current,
            baseline_value: baseline,
            change_percent: change,
            regression_detected: regression,
            severity,
        }
    }
}

/// Severity levels for regressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Stable,
    Improvement,
    Warning,
    Critical,
}

impl std::fmt::Display for RegressionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionSeverity::Stable => write!(f, "STABLE"),
            RegressionSeverity::Improvement => write!(f, "IMPROVED"),
            RegressionSeverity::Warning => write!(f, "WARNING"),
            RegressionSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Complete regression report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub timestamp: String,
    pub baseline_timestamp: Option<String>,
    pub total_benchmarks: usize,
    pub regressions: usize,
    pub improvements: usize,
    pub stable: usize,
    pub critical: usize,
    pub results: Vec<RegressionResult>,
    pub overhead_status: OverheadStatus,
    pub passed: bool,
    pub exit_code: i32,
    pub summary_message: String,
}

/// Overhead status summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadStatus {
    pub avg_overhead: f64,
    pub max_overhead: f64,
    pub meets_target: bool,
    pub exceeds_alert: bool,
    pub benchmarks_meeting_target: usize,
    pub benchmarks_exceeding_alert: usize,
}

/// Baseline data storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub timestamp: String,
    pub git_commit: Option<String>,
    pub measurements: BTreeMap<String, f64>,
    pub overhead_values: BTreeMap<String, f64>,
}

impl Baseline {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: get_git_commit(),
            measurements: BTreeMap::new(),
            overhead_values: BTreeMap::new(),
        }
    }

    pub fn add_measurement(&mut self, name: &str, value: f64) {
        self.measurements.insert(name.to_string(), value);
    }

    pub fn add_overhead(&mut self, name: &str, overhead: f64) {
        self.overhead_values.insert(name.to_string(), overhead);
    }

    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    pub fn load(path: &PathBuf) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let baseline: Self = serde_json::from_str(&json)?;
        Ok(baseline)
    }
}

impl Default for Baseline {
    fn default() -> Self {
        Self::new()
    }
}

/// Regression detector.
pub struct RegressionDetector {
    baseline: Option<Baseline>,
    threshold_percent: f64,
    output_dir: PathBuf,
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self {
            baseline: None,
            threshold_percent: MAX_REGRESSION_PERCENT,
            output_dir: PathBuf::from("target/helix-benchmarks"),
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold_percent = threshold;
        self
    }

    pub fn with_output_dir(mut self, dir: PathBuf) -> Self {
        self.output_dir = dir;
        self
    }

    /// Loads baseline from the default location.
    pub fn load_baseline(&mut self) -> bool {
        let baseline_path = self.output_dir.join("baseline.json");
        if let Ok(baseline) = Baseline::load(&baseline_path) {
            self.baseline = Some(baseline);
            true
        } else {
            false
        }
    }

    /// Loads baseline from a specific path.
    pub fn load_baseline_from(&mut self, path: &PathBuf) -> bool {
        if let Ok(baseline) = Baseline::load(path) {
            self.baseline = Some(baseline);
            true
        } else {
            false
        }
    }

    /// Compares a benchmark report against the baseline.
    pub fn compare(&self, report: &BenchmarkReport) -> RegressionReport {
        let mut results = Vec::new();
        let mut regressions = 0;
        let mut improvements = 0;
        let mut stable = 0;
        let mut critical = 0;

        for result in &report.results {
            let baseline_value = self
                .baseline
                .as_ref()
                .and_then(|b| b.measurements.get(&result.name))
                .copied()
                .unwrap_or(result.metrics.mean_ns);

            let reg_result = RegressionResult::new(
                &result.name,
                result.metrics.mean_ns,
                baseline_value,
                self.threshold_percent,
            );

            match reg_result.severity {
                RegressionSeverity::Critical => {
                    critical += 1;
                    regressions += 1;
                }
                RegressionSeverity::Warning => {
                    regressions += 1;
                }
                RegressionSeverity::Improvement => {
                    improvements += 1;
                }
                RegressionSeverity::Stable => {
                    stable += 1;
                }
            }

            results.push(reg_result);
        }

        // Calculate overhead status
        let overhead_values: Vec<f64> = self
            .baseline
            .as_ref()
            .map(|b| b.overhead_values.values().cloned().collect())
            .unwrap_or_default();

        let overhead_status = OverheadStatus {
            avg_overhead: if overhead_values.is_empty() {
                0.0
            } else {
                overhead_values.iter().sum::<f64>() / overhead_values.len() as f64
            },
            max_overhead: overhead_values.iter().cloned().fold(0.0f64, f64::max),
            meets_target: overhead_values.iter().all(|&o| o <= TARGET_OVERHEAD_MULTIPLE),
            exceeds_alert: overhead_values.iter().any(|&o| o > REGRESSION_ALERT_THRESHOLD),
            benchmarks_meeting_target: overhead_values
                .iter()
                .filter(|&&o| o <= TARGET_OVERHEAD_MULTIPLE)
                .count(),
            benchmarks_exceeding_alert: overhead_values
                .iter()
                .filter(|&&o| o > REGRESSION_ALERT_THRESHOLD)
                .count(),
        };

        // Determine overall status
        let passed = regressions == 0 && !overhead_status.exceeds_alert;
        let exit_code = if critical > 0 || overhead_status.exceeds_alert {
            2
        } else if regressions > 0 || !overhead_status.meets_target {
            1
        } else {
            0
        };

        let summary_message = if critical > 0 {
            format!(
                "CRITICAL: {} critical regressions detected",
                critical
            )
        } else if regressions > 0 {
            format!(
                "{} regressions detected (threshold: {}%)",
                regressions, self.threshold_percent
            )
        } else if overhead_status.exceeds_alert {
            format!(
                "ALERT: {} benchmarks exceed {:.0}x threshold",
                overhead_status.benchmarks_exceeding_alert, REGRESSION_ALERT_THRESHOLD
            )
        } else if !overhead_status.meets_target {
            format!(
                "WARNING: {} benchmarks exceed {:.0}x target",
                overhead_values.len() - overhead_status.benchmarks_meeting_target,
                TARGET_OVERHEAD_MULTIPLE
            )
        } else {
            "All benchmarks passed".to_string()
        };

        RegressionReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            baseline_timestamp: self.baseline.as_ref().map(|b| b.timestamp.clone()),
            total_benchmarks: report.results.len(),
            regressions,
            improvements,
            stable,
            critical,
            results,
            overhead_status,
            passed,
            exit_code,
            summary_message,
        }
    }

    /// Compares overhead values directly.
    pub fn compare_overheads(&self, overheads: &BTreeMap<String, f64>) -> RegressionReport {
        let mut results = Vec::new();
        let mut regressions = 0;
        let mut improvements = 0;
        let mut stable = 0;
        let mut critical = 0;

        for (name, &current) in overheads {
            let baseline_value = self
                .baseline
                .as_ref()
                .and_then(|b| b.overhead_values.get(name))
                .copied()
                .unwrap_or(current);

            let reg_result = RegressionResult::new(
                name,
                current,
                baseline_value,
                self.threshold_percent,
            );

            match reg_result.severity {
                RegressionSeverity::Critical => {
                    critical += 1;
                    regressions += 1;
                }
                RegressionSeverity::Warning => {
                    regressions += 1;
                }
                RegressionSeverity::Improvement => {
                    improvements += 1;
                }
                RegressionSeverity::Stable => {
                    stable += 1;
                }
            }

            results.push(reg_result);
        }

        // Check overhead thresholds
        let exceeds_target = overheads.values().any(|&o| o > TARGET_OVERHEAD_MULTIPLE);
        let exceeds_alert = overheads.values().any(|&o| o > REGRESSION_ALERT_THRESHOLD);
        let avg_overhead: f64 = if overheads.is_empty() {
            0.0
        } else {
            overheads.values().sum::<f64>() / overheads.len() as f64
        };
        let max_overhead = overheads.values().cloned().fold(0.0f64, f64::max);

        let overhead_status = OverheadStatus {
            avg_overhead,
            max_overhead,
            meets_target: !exceeds_target,
            exceeds_alert,
            benchmarks_meeting_target: overheads
                .values()
                .filter(|&&o| o <= TARGET_OVERHEAD_MULTIPLE)
                .count(),
            benchmarks_exceeding_alert: overheads
                .values()
                .filter(|&&o| o > REGRESSION_ALERT_THRESHOLD)
                .count(),
        };

        let passed = regressions == 0 && !exceeds_alert;
        let exit_code = if critical > 0 || exceeds_alert {
            2
        } else if regressions > 0 || exceeds_target {
            1
        } else {
            0
        };

        let summary_message = if exceeds_alert {
            format!(
                "ALERT: {} benchmarks exceed {:.0}x alert threshold (max: {:.1}x)",
                overhead_status.benchmarks_exceeding_alert, REGRESSION_ALERT_THRESHOLD, max_overhead
            )
        } else if exceeds_target {
            format!(
                "WARNING: {} benchmarks exceed {:.0}x target (max: {:.1}x)",
                overheads.len() - overhead_status.benchmarks_meeting_target,
                TARGET_OVERHEAD_MULTIPLE,
                max_overhead
            )
        } else if regressions > 0 {
            format!(
                "{} regressions vs baseline (threshold: {}%)",
                regressions, self.threshold_percent
            )
        } else {
            format!(
                "All {} benchmarks passed (avg overhead: {:.1}x)",
                overheads.len(),
                avg_overhead
            )
        };

        RegressionReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            baseline_timestamp: self.baseline.as_ref().map(|b| b.timestamp.clone()),
            total_benchmarks: overheads.len(),
            regressions,
            improvements,
            stable,
            critical,
            results,
            overhead_status,
            passed,
            exit_code,
            summary_message,
        }
    }

    /// Saves current results as the new baseline.
    pub fn save_baseline(&self, overheads: &BTreeMap<String, f64>) -> std::io::Result<PathBuf> {
        let mut baseline = Baseline::new();
        for (name, &overhead) in overheads {
            baseline.add_overhead(name, overhead);
        }

        let path = self.output_dir.join("baseline.json");
        baseline.save(&path)?;
        Ok(path)
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn get_git_commit() -> Option<String> {
    std::env::var("GITHUB_SHA").ok().or_else(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    })
}

/// Generates a markdown regression report.
pub fn generate_regression_markdown(report: &RegressionReport) -> String {
    let mut md = String::new();

    md.push_str("# HELIX Regression Detection Report\n\n");
    md.push_str(&format!("**Generated:** {}\n", report.timestamp));
    if let Some(ref baseline) = report.baseline_timestamp {
        md.push_str(&format!("**Baseline:** {}\n", baseline));
    }
    md.push_str("\n");

    // Summary
    md.push_str("## Summary\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!(
        "| Total Benchmarks | {} |\n",
        report.total_benchmarks
    ));
    md.push_str(&format!(
        "| Regressions | {} {} |\n",
        report.regressions,
        if report.regressions > 0 { "🔴" } else { "" }
    ));
    md.push_str(&format!(
        "| Critical | {} {} |\n",
        report.critical,
        if report.critical > 0 { "🚨" } else { "" }
    ));
    md.push_str(&format!(
        "| Improvements | {} {} |\n",
        report.improvements,
        if report.improvements > 0 { "🟢" } else { "" }
    ));
    md.push_str(&format!("| Stable | {} |\n", report.stable));
    md.push_str(&format!(
        "| Status | {} |\n\n",
        if report.passed { "✅ PASSED" } else { "❌ FAILED" }
    ));

    // Overhead status
    md.push_str("## Overhead Status\n\n");
    md.push_str(&format!(
        "- **Average Overhead:** {:.1}x (target: ≤{:.0}x)\n",
        report.overhead_status.avg_overhead, TARGET_OVERHEAD_MULTIPLE
    ));
    md.push_str(&format!(
        "- **Maximum Overhead:** {:.1}x (alert: >{:.0}x)\n",
        report.overhead_status.max_overhead, REGRESSION_ALERT_THRESHOLD
    ));
    md.push_str(&format!(
        "- **Meeting Target:** {}/{}\n",
        report.overhead_status.benchmarks_meeting_target, report.total_benchmarks
    ));
    md.push_str(&format!(
        "- **Exceeding Alert:** {}\n\n",
        report.overhead_status.benchmarks_exceeding_alert
    ));

    // Detailed results
    if !report.results.is_empty() {
        md.push_str("## Detailed Results\n\n");
        md.push_str("| Benchmark | Current | Baseline | Change | Severity |\n");
        md.push_str("|-----------|---------|----------|--------|----------|\n");

        for result in &report.results {
            let icon = match result.severity {
                RegressionSeverity::Critical => "🚨",
                RegressionSeverity::Warning => "⚠️",
                RegressionSeverity::Improvement => "✅",
                RegressionSeverity::Stable => "➖",
            };

            md.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:+.1}% | {} {} |\n",
                result.benchmark_name,
                result.current_value,
                result.baseline_value,
                result.change_percent,
                result.severity,
                icon,
            ));
        }
    }

    md.push_str("\n## Result\n\n");
    md.push_str(&format!("**{}**\n\n", report.summary_message));
    md.push_str(&format!("Exit code: {}\n", report.exit_code));

    md
}

/// Prints regression report to terminal.
pub fn print_regression_terminal(report: &RegressionReport) {
    println!("\n{}", "=".repeat(70));
    println!("HELIX Regression Detection Report");
    println!("{}", "=".repeat(70));
    println!("Generated: {}", report.timestamp);
    if let Some(ref baseline) = report.baseline_timestamp {
        println!("Baseline: {}", baseline);
    }
    println!();

    println!("Summary:");
    println!("  Total benchmarks: {}", report.total_benchmarks);
    println!(
        "  Regressions: {} {}",
        report.regressions,
        if report.regressions > 0 { "(!!)" } else { "" }
    );
    println!(
        "  Critical: {} {}",
        report.critical,
        if report.critical > 0 { "(!!!)" } else { "" }
    );
    println!("  Improvements: {}", report.improvements);
    println!("  Stable: {}", report.stable);
    println!();

    println!("Overhead Status:");
    println!(
        "  Average: {:.1}x (target: ≤{:.0}x)",
        report.overhead_status.avg_overhead, TARGET_OVERHEAD_MULTIPLE
    );
    println!(
        "  Maximum: {:.1}x (alert: >{:.0}x)",
        report.overhead_status.max_overhead, REGRESSION_ALERT_THRESHOLD
    );
    println!(
        "  Meeting target: {}/{}",
        report.overhead_status.benchmarks_meeting_target, report.total_benchmarks
    );
    println!();

    // Show any regressions
    let regressions: Vec<_> = report
        .results
        .iter()
        .filter(|r| r.regression_detected)
        .collect();

    if !regressions.is_empty() {
        println!("Regressions detected:");
        for reg in &regressions {
            println!(
                "  {} [{:?}]: {:.2} -> {:.2} ({:+.1}%)",
                reg.benchmark_name,
                reg.severity,
                reg.baseline_value,
                reg.current_value,
                reg.change_percent
            );
        }
        println!();
    }

    println!("{}", "=".repeat(70));
    println!(
        "STATUS: {}",
        if report.passed { "PASSED" } else { "FAILED" }
    );
    println!("{}", report.summary_message);
    println!("Exit code: {}", report.exit_code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_detection() {
        let detector = RegressionDetector::new().with_threshold(10.0);

        let mut overheads = BTreeMap::new();
        overheads.insert("test_1".to_string(), 15.0);
        overheads.insert("test_2".to_string(), 20.0);

        let report = detector.compare_overheads(&overheads);

        assert!(report.passed);
        assert!(report.overhead_status.meets_target);
    }

    #[test]
    fn test_regression_exceeds_target() {
        let detector = RegressionDetector::new();

        let mut overheads = BTreeMap::new();
        overheads.insert("test_1".to_string(), 32.0); // Exceeds 30x target but not 35x alert

        let report = detector.compare_overheads(&overheads);

        // Overhead exceeds target (30x) but not alert (35x)
        assert!(!report.overhead_status.meets_target);
        // Still passes because it doesn't exceed alert and no regressions
        assert!(report.passed);
        // But exit code should be 1 (target exceeded)
        assert_eq!(report.exit_code, 1);
    }

    #[test]
    fn test_regression_exceeds_alert() {
        let detector = RegressionDetector::new();

        let mut overheads = BTreeMap::new();
        overheads.insert("test_1".to_string(), 40.0); // Exceeds 35x alert

        let report = detector.compare_overheads(&overheads);

        assert!(report.overhead_status.exceeds_alert);
        assert_eq!(report.exit_code, 2);
    }

    #[test]
    fn test_baseline_save_load() {
        let mut baseline = Baseline::new();
        baseline.add_overhead("test_1", 15.0);
        baseline.add_overhead("test_2", 20.0);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("helix_test_baseline.json");

        baseline.save(&path).expect("Failed to save baseline");
        let loaded = Baseline::load(&path).expect("Failed to load baseline");

        assert_eq!(loaded.overhead_values.get("test_1"), Some(&15.0));
        assert_eq!(loaded.overhead_values.get("test_2"), Some(&20.0));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
