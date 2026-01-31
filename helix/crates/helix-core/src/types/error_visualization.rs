//! Error bound visualization serialization for dashboard.
//!
//! This module provides serialization formats for error bounds that can be
//! consumed by web dashboards and visualization tools.

use super::probabilistic_error::{ConfidenceLevel, ProbabilisticError};
use super::precision::Precision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serializable error snapshot for a single value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSnapshot {
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Step number (training iteration).
    pub step: usize,
    /// Computed value.
    pub value: f64,
    /// Absolute error bound.
    pub absolute_error: f64,
    /// Relative error bound (if applicable).
    pub relative_error: Option<f64>,
    /// Lower bound of confidence interval.
    pub ci_lower: f64,
    /// Upper bound of confidence interval.
    pub ci_upper: f64,
    /// Confidence level (0-1).
    pub confidence: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Precision used.
    pub precision: String,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ErrorSnapshot {
    /// Creates a new error snapshot.
    pub fn new(
        timestamp_ms: u64,
        step: usize,
        value: f64,
        error: &ProbabilisticError,
        precision: Precision,
    ) -> Self {
        let ci = error.confidence_interval(value, ConfidenceLevel::NinetyFive);

        Self {
            timestamp_ms,
            step,
            value,
            absolute_error: error.worst_case,
            relative_error: if value.abs() > f64::EPSILON {
                Some(error.worst_case / value.abs())
            } else {
                None
            },
            ci_lower: ci.lower,
            ci_upper: ci.upper,
            confidence: ci.confidence,
            std_dev: error.std_dev,
            precision: precision.name().to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to the snapshot.
    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Converts to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Time series of error snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTimeSeries {
    /// Series name.
    pub name: String,
    /// Series description.
    pub description: Option<String>,
    /// Data points.
    pub data: Vec<ErrorSnapshot>,
    /// Statistics over the series.
    pub statistics: Option<SeriesStatistics>,
}

/// Statistics for a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStatistics {
    pub count: usize,
    pub mean_value: f64,
    pub mean_error: f64,
    pub max_error: f64,
    pub min_error: f64,
    pub error_trend: f64,
    pub error_variance: f64,
}

impl ErrorTimeSeries {
    /// Creates a new time series.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            data: Vec::new(),
            statistics: None,
        }
    }

    /// Adds a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Adds a data point.
    pub fn add(&mut self, snapshot: ErrorSnapshot) {
        self.data.push(snapshot);
    }

    /// Computes and stores statistics.
    pub fn compute_statistics(&mut self) {
        if self.data.is_empty() {
            return;
        }

        let count = self.data.len();
        let mean_value = self.data.iter().map(|s| s.value).sum::<f64>() / count as f64;
        let mean_error = self.data.iter().map(|s| s.absolute_error).sum::<f64>() / count as f64;
        let max_error = self
            .data
            .iter()
            .map(|s| s.absolute_error)
            .fold(0.0, f64::max);
        let min_error = self
            .data
            .iter()
            .map(|s| s.absolute_error)
            .fold(f64::INFINITY, f64::min);

        // Compute trend (linear regression slope)
        let n = count as f64;
        let sum_x: f64 = (0..count).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().map(|s| s.absolute_error).sum();
        let sum_xy: f64 = self
            .data
            .iter()
            .enumerate()
            .map(|(i, s)| i as f64 * s.absolute_error)
            .sum();
        let sum_x2: f64 = (0..count).map(|i| (i * i) as f64).sum();

        let error_trend = if n * sum_x2 - sum_x * sum_x != 0.0 {
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        } else {
            0.0
        };

        let error_variance = self
            .data
            .iter()
            .map(|s| (s.absolute_error - mean_error).powi(2))
            .sum::<f64>()
            / count as f64;

        self.statistics = Some(SeriesStatistics {
            count,
            mean_value,
            mean_error,
            max_error,
            min_error,
            error_trend,
            error_variance,
        });
    }

    /// Converts to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Converts to JSON with pretty printing.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Dashboard state for error visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDashboardState {
    /// Current training step.
    pub current_step: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Error budget used (0-1).
    pub budget_used: f64,
    /// Error budget remaining.
    pub budget_remaining: f64,
    /// Current precision.
    pub current_precision: String,
    /// Current phase.
    pub current_phase: String,
    /// Active workers (for distributed training).
    pub active_workers: usize,
    /// Time series for different metrics.
    pub series: HashMap<String, ErrorTimeSeries>,
    /// Alerts/warnings.
    pub alerts: Vec<Alert>,
    /// Summary metrics.
    pub summary: DashboardSummary,
}

/// An alert for the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert level.
    pub level: AlertLevel,
    /// Alert message.
    pub message: String,
    /// Timestamp.
    pub timestamp_ms: u64,
    /// Step when alert occurred.
    pub step: usize,
}

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Summary metrics for dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Average error over training.
    pub average_error: f64,
    /// Maximum error observed.
    pub max_error: f64,
    /// Error trend (positive = increasing).
    pub error_trend: f64,
    /// Estimated time remaining.
    pub estimated_time_remaining_ms: Option<u64>,
    /// Proof generation stats.
    pub proof_stats: Option<ProofStats>,
}

/// Proof generation statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStats {
    /// Total proofs generated.
    pub total_generated: usize,
    /// Total proofs verified.
    pub total_verified: usize,
    /// Average proof generation time (ms).
    pub avg_prove_time_ms: f64,
    /// Average verification time (ms).
    pub avg_verify_time_ms: f64,
    /// Total proof size (bytes).
    pub total_proof_size_bytes: usize,
}

impl ErrorDashboardState {
    /// Creates a new dashboard state.
    pub fn new(total_steps: usize, total_budget: f64) -> Self {
        Self {
            current_step: 0,
            total_steps,
            budget_used: 0.0,
            budget_remaining: total_budget,
            current_precision: "fp32".to_string(),
            current_phase: "warmup".to_string(),
            active_workers: 0,
            series: HashMap::new(),
            alerts: Vec::new(),
            summary: DashboardSummary {
                average_error: 0.0,
                max_error: 0.0,
                error_trend: 0.0,
                estimated_time_remaining_ms: None,
                proof_stats: None,
            },
        }
    }

    /// Adds or updates a time series.
    pub fn add_series(&mut self, name: &str, series: ErrorTimeSeries) {
        self.series.insert(name.to_string(), series);
    }

    /// Adds an alert.
    pub fn add_alert(&mut self, level: AlertLevel, message: &str, step: usize) {
        self.alerts.push(Alert {
            level,
            message: message.to_string(),
            timestamp_ms: current_timestamp_ms(),
            step,
        });

        // Keep only last 100 alerts
        if self.alerts.len() > 100 {
            self.alerts.remove(0);
        }
    }

    /// Updates the current step.
    pub fn update_step(&mut self, step: usize, error: f64, precision: &str, phase: &str) {
        self.current_step = step;
        self.budget_used += error;
        self.budget_remaining -= error;
        self.current_precision = precision.to_string();
        self.current_phase = phase.to_string();

        // Update summary
        if error > self.summary.max_error {
            self.summary.max_error = error;
        }

        // Running average
        if step > 0 {
            self.summary.average_error =
                (self.summary.average_error * (step - 1) as f64 + error) / step as f64;
        }
    }

    /// Converts to JSON for transmission to dashboard.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Converts to a compact JSON format for frequent updates.
    pub fn to_compact_json(&self) -> String {
        // Only include essential fields for quick updates
        serde_json::json!({
            "step": self.current_step,
            "total_steps": self.total_steps,
            "budget_used": self.budget_used,
            "precision": self.current_precision,
            "phase": self.current_phase,
            "avg_error": self.summary.average_error,
            "max_error": self.summary.max_error,
            "alerts_count": self.alerts.len(),
        })
        .to_string()
    }
}

/// Chart data format for web visualizations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    /// Chart type.
    pub chart_type: ChartType,
    /// Chart title.
    pub title: String,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Data series.
    pub series: Vec<ChartSeries>,
    /// Annotations (vertical lines, regions, etc.).
    #[serde(default)]
    pub annotations: Vec<ChartAnnotation>,
}

/// Type of chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChartType {
    Line,
    Area,
    Bar,
    Scatter,
    Heatmap,
}

/// A data series for a chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSeries {
    /// Series name.
    pub name: String,
    /// X values.
    pub x: Vec<f64>,
    /// Y values.
    pub y: Vec<f64>,
    /// Optional error bars (lower, upper).
    pub error_bars: Option<(Vec<f64>, Vec<f64>)>,
    /// Series color.
    pub color: Option<String>,
    /// Line style.
    pub style: Option<String>,
}

impl ChartSeries {
    /// Creates a new chart series.
    pub fn new(name: &str, x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            x,
            y,
            error_bars: None,
            color: None,
            style: None,
        }
    }

    /// Adds error bars.
    pub fn with_error_bars(mut self, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        self.error_bars = Some((lower, upper));
        self
    }

    /// Sets the color.
    pub fn with_color(mut self, color: &str) -> Self {
        self.color = Some(color.to_string());
        self
    }
}

/// Chart annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAnnotation {
    /// Annotation type.
    pub annotation_type: AnnotationType,
    /// X position (or start for regions).
    pub x: f64,
    /// X end (for regions).
    pub x_end: Option<f64>,
    /// Label text.
    pub label: String,
    /// Color.
    pub color: Option<String>,
}

/// Type of annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnnotationType {
    VerticalLine,
    Region,
    Point,
    Text,
}

impl ChartData {
    /// Creates a new chart.
    pub fn new(chart_type: ChartType, title: &str, x_label: &str, y_label: &str) -> Self {
        Self {
            chart_type,
            title: title.to_string(),
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
            series: Vec::new(),
            annotations: Vec::new(),
        }
    }

    /// Adds a series.
    pub fn add_series(&mut self, series: ChartSeries) {
        self.series.push(series);
    }

    /// Adds an annotation.
    pub fn add_annotation(&mut self, annotation: ChartAnnotation) {
        self.annotations.push(annotation);
    }

    /// Converts to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Creates an error trend chart from time series data.
pub fn create_error_trend_chart(series: &ErrorTimeSeries) -> ChartData {
    let mut chart = ChartData::new(
        ChartType::Line,
        &format!("Error Trend: {}", series.name),
        "Step",
        "Error",
    );

    let x: Vec<f64> = series.data.iter().map(|s| s.step as f64).collect();
    let y: Vec<f64> = series.data.iter().map(|s| s.absolute_error).collect();
    let lower: Vec<f64> = series.data.iter().map(|s| s.ci_lower).collect();
    let upper: Vec<f64> = series.data.iter().map(|s| s.ci_upper).collect();

    // Main error line
    let main_series = ChartSeries::new("Absolute Error", x.clone(), y).with_color("#2196F3");
    chart.add_series(main_series);

    // Confidence interval bounds
    let ci_series = ChartSeries::new("95% CI", x.clone(), lower.clone())
        .with_error_bars(lower, upper)
        .with_color("#90CAF9");
    chart.add_series(ci_series);

    chart
}

/// Creates a precision distribution chart.
pub fn create_precision_distribution_chart(
    precision_counts: &HashMap<String, usize>,
) -> ChartData {
    let mut chart = ChartData::new(
        ChartType::Bar,
        "Precision Distribution",
        "Precision",
        "Count",
    );

    let x: Vec<f64> = (0..precision_counts.len()).map(|i| i as f64).collect();
    let y: Vec<f64> = precision_counts.values().map(|&v| v as f64).collect();

    let series = ChartSeries::new("Precision Usage", x, y).with_color("#4CAF50");
    chart.add_series(series);

    chart
}

/// Creates an error budget gauge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeData {
    /// Gauge title.
    pub title: String,
    /// Current value.
    pub value: f64,
    /// Maximum value.
    pub max_value: f64,
    /// Threshold zones.
    pub zones: Vec<GaugeZone>,
}

/// A zone on a gauge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeZone {
    /// Zone start (0-1).
    pub start: f64,
    /// Zone end (0-1).
    pub end: f64,
    /// Zone color.
    pub color: String,
}

impl GaugeData {
    /// Creates an error budget gauge.
    pub fn error_budget(used: f64, total: f64) -> Self {
        Self {
            title: "Error Budget".to_string(),
            value: used,
            max_value: total,
            zones: vec![
                GaugeZone {
                    start: 0.0,
                    end: 0.5,
                    color: "#4CAF50".to_string(), // Green
                },
                GaugeZone {
                    start: 0.5,
                    end: 0.8,
                    color: "#FFC107".to_string(), // Yellow
                },
                GaugeZone {
                    start: 0.8,
                    end: 1.0,
                    color: "#F44336".to_string(), // Red
                },
            ],
        }
    }

    /// Converts to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Helper to get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// SSE (Server-Sent Events) message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseMessage {
    /// Event type.
    pub event: String,
    /// Event data (JSON string).
    pub data: String,
    /// Optional event ID.
    pub id: Option<String>,
}

impl SseMessage {
    /// Creates a new SSE message.
    pub fn new(event: &str, data: &str) -> Self {
        Self {
            event: event.to_string(),
            data: data.to_string(),
            id: None,
        }
    }

    /// Adds an ID.
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// Formats as SSE text.
    pub fn to_sse(&self) -> String {
        let mut result = String::new();

        if let Some(id) = &self.id {
            result.push_str(&format!("id: {}\n", id));
        }

        result.push_str(&format!("event: {}\n", self.event));
        result.push_str(&format!("data: {}\n\n", self.data));

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_snapshot() {
        let error = ProbabilisticError::from_absolute(0.01);
        let snapshot = ErrorSnapshot::new(1000, 10, 1.5, &error, Precision::F32);

        assert_eq!(snapshot.step, 10);
        assert!((snapshot.absolute_error - 0.01).abs() < 0.001);

        let json = snapshot.to_json();
        assert!(json.contains("\"step\":10"));
    }

    #[test]
    fn test_error_time_series() {
        let mut series = ErrorTimeSeries::new("loss_error");

        for i in 0..10 {
            let error = ProbabilisticError::from_absolute(0.01 - i as f64 * 0.001);
            let snapshot = ErrorSnapshot::new(i as u64 * 1000, i, 1.0, &error, Precision::F32);
            series.add(snapshot);
        }

        series.compute_statistics();

        let stats = series.statistics.as_ref().unwrap();
        assert_eq!(stats.count, 10);
        assert!(stats.error_trend < 0.0); // Decreasing error
    }

    #[test]
    fn test_dashboard_state() {
        let mut state = ErrorDashboardState::new(1000, 0.1);

        state.update_step(1, 0.001, "bf16", "warmup");
        state.add_alert(AlertLevel::Info, "Training started", 0);

        let json = state.to_json();
        assert!(json.contains("\"current_step\":1"));

        let compact = state.to_compact_json();
        assert!(compact.contains("\"step\":1"));
    }

    #[test]
    fn test_chart_data() {
        let mut chart = ChartData::new(ChartType::Line, "Error Over Time", "Step", "Error");

        let series = ChartSeries::new("error", vec![1.0, 2.0, 3.0], vec![0.1, 0.05, 0.03])
            .with_color("#ff0000");

        chart.add_series(series);

        let json = chart.to_json();
        assert!(json.contains("\"chart_type\":\"line\""));
        assert!(json.contains("Error Over Time"));
    }

    #[test]
    fn test_gauge_data() {
        let gauge = GaugeData::error_budget(0.05, 0.1);

        assert_eq!(gauge.zones.len(), 3);
        assert_eq!(gauge.value, 0.05);
    }

    #[test]
    fn test_sse_message() {
        let msg = SseMessage::new("error_update", r#"{"error": 0.01}"#).with_id("123");

        let sse = msg.to_sse();
        assert!(sse.contains("id: 123"));
        assert!(sse.contains("event: error_update"));
        assert!(sse.contains(r#"data: {"error": 0.01}"#));
    }
}
