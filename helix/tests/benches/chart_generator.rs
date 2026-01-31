//! Publication-Quality Chart Generation
//!
//! Generates SVG charts and ASCII tables suitable for publication and
//! documentation. These visualizations show:
//!
//! - Overhead vs model size plots
//! - Native vs proof time comparisons
//! - Scaling behavior analysis
//! - GKR vs Halo2 comparisons
//!
//! # Output Formats
//!
//! - SVG for vector graphics (embedding in papers/docs)
//! - ASCII tables for terminal and markdown
//! - CSV for data export
//!
//! # Usage
//!
//! ```rust,ignore
//! use helix_integration_tests::benches::chart_generator::ChartGenerator;
//!
//! let generator = ChartGenerator::new();
//! generator.generate_overhead_chart(&data, "target/reports/overhead.svg");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use super::{
    overhead_report::OverheadReport, scaling::ScalingAnalysis, ModelSize,
    REGRESSION_ALERT_THRESHOLD, TARGET_OVERHEAD_MULTIPLE,
};

/// Chart color palette for consistent styling.
pub struct ChartColors;

impl ChartColors {
    pub const NATIVE: &'static str = "#2563eb"; // Blue
    pub const GKR: &'static str = "#16a34a"; // Green
    pub const HALO2: &'static str = "#dc2626"; // Red
    pub const TARGET_LINE: &'static str = "#f59e0b"; // Amber
    pub const ALERT_LINE: &'static str = "#ef4444"; // Red
    pub const GRID: &'static str = "#e5e7eb"; // Gray
    pub const TEXT: &'static str = "#1f2937"; // Dark gray
    pub const BACKGROUND: &'static str = "#ffffff"; // White
}

/// Data point for chart generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataPoint {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub category: String,
}

/// Chart configuration options.
#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub width: usize,
    pub height: usize,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub show_legend: bool,
    pub show_grid: bool,
    pub log_x: bool,
    pub log_y: bool,
    pub show_target_line: bool,
    pub show_alert_line: bool,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 500,
            title: "HELIX Benchmark Results".to_string(),
            x_label: "Model Size (parameters)".to_string(),
            y_label: "Overhead (x)".to_string(),
            show_legend: true,
            show_grid: true,
            log_x: true,
            log_y: false,
            show_target_line: true,
            show_alert_line: true,
        }
    }
}

impl ChartConfig {
    pub fn overhead_chart() -> Self {
        Self {
            title: "HELIX Overhead vs Model Size".to_string(),
            x_label: "Model Parameters".to_string(),
            y_label: "Overhead Multiple (proof/native)".to_string(),
            ..Default::default()
        }
    }

    pub fn timing_chart() -> Self {
        Self {
            title: "HELIX Proof Generation Time".to_string(),
            x_label: "Model Parameters".to_string(),
            y_label: "Time (ms)".to_string(),
            log_y: true,
            show_target_line: false,
            show_alert_line: false,
            ..Default::default()
        }
    }

    pub fn scaling_chart() -> Self {
        Self {
            title: "HELIX Scaling Analysis".to_string(),
            x_label: "Input Size".to_string(),
            y_label: "Overhead Multiple".to_string(),
            log_x: true,
            ..Default::default()
        }
    }
}

/// Generator for publication-quality charts.
pub struct ChartGenerator {
    output_dir: PathBuf,
}

impl ChartGenerator {
    /// Creates a new chart generator with the default output directory.
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("target/helix-benchmarks/charts"),
        }
    }

    /// Creates a chart generator with a custom output directory.
    pub fn with_output_dir(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }

    /// Ensures the output directory exists.
    fn ensure_output_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.output_dir)
    }

    /// Generates an SVG overhead chart.
    pub fn generate_overhead_svg(
        &self,
        report: &OverheadReport,
        filename: &str,
    ) -> std::io::Result<PathBuf> {
        self.ensure_output_dir()?;
        let path = self.output_dir.join(filename);
        let config = ChartConfig::overhead_chart();

        let mut data_points = Vec::new();
        for (name, model_data) in &report.by_model_size {
            if let Some(ref gkr) = model_data.gkr_overhead {
                data_points.push(ChartDataPoint {
                    x: model_data.param_count as f64,
                    y: gkr.overhead,
                    label: name.clone(),
                    category: "GKR".to_string(),
                });
            }
            if let Some(ref halo2) = model_data.halo2_overhead {
                data_points.push(ChartDataPoint {
                    x: model_data.param_count as f64,
                    y: halo2.overhead,
                    label: name.clone(),
                    category: "Halo2".to_string(),
                });
            }
        }

        let svg = self.render_svg(&data_points, &config);
        std::fs::write(&path, svg)?;
        Ok(path)
    }

    /// Generates an SVG timing chart.
    pub fn generate_timing_svg(
        &self,
        report: &OverheadReport,
        filename: &str,
    ) -> std::io::Result<PathBuf> {
        self.ensure_output_dir()?;
        let path = self.output_dir.join(filename);
        let config = ChartConfig::timing_chart();

        let mut data_points = Vec::new();
        for (name, model_data) in &report.by_model_size {
            if let Some(ref gkr) = model_data.gkr_overhead {
                data_points.push(ChartDataPoint {
                    x: model_data.param_count as f64,
                    y: gkr.native_time.as_secs_f64() * 1000.0,
                    label: format!("{}_native", name),
                    category: "Native".to_string(),
                });
                data_points.push(ChartDataPoint {
                    x: model_data.param_count as f64,
                    y: gkr.proof_time.as_secs_f64() * 1000.0,
                    label: format!("{}_gkr", name),
                    category: "GKR".to_string(),
                });
            }
        }

        let svg = self.render_svg(&data_points, &config);
        std::fs::write(&path, svg)?;
        Ok(path)
    }

    /// Generates an SVG scaling analysis chart.
    pub fn generate_scaling_svg(
        &self,
        analysis: &ScalingAnalysis,
        filename: &str,
    ) -> std::io::Result<PathBuf> {
        self.ensure_output_dir()?;
        let path = self.output_dir.join(filename);
        let config = ChartConfig::scaling_chart();

        let data_points: Vec<ChartDataPoint> = analysis
            .points
            .iter()
            .map(|p| ChartDataPoint {
                x: p.size as f64,
                y: p.overhead,
                label: p.size.to_string(),
                category: "Overhead".to_string(),
            })
            .collect();

        let svg = self.render_svg(&data_points, &config);
        std::fs::write(&path, svg)?;
        Ok(path)
    }

    /// Renders data points to an SVG string.
    fn render_svg(&self, data: &[ChartDataPoint], config: &ChartConfig) -> String {
        let margin = 60.0;
        let plot_width = config.width as f64 - 2.0 * margin;
        let plot_height = config.height as f64 - 2.0 * margin - 40.0; // Extra for title

        // Calculate data ranges
        let (x_min, x_max, y_min, y_max) = self.calculate_ranges(data, config);

        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            config.width, config.height, config.width, config.height
        ));
        svg.push('\n');

        // Background
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="{}"/>"#,
            ChartColors::BACKGROUND
        ));
        svg.push('\n');

        // Title
        svg.push_str(&format!(
            r#"  <text x="{}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
            config.width / 2,
            ChartColors::TEXT,
            config.title
        ));
        svg.push('\n');

        // Plot area background
        svg.push_str(&format!(
            r#"  <rect x="{}" y="50" width="{}" height="{}" fill="none" stroke="{}" stroke-width="1"/>"#,
            margin, plot_width, plot_height, ChartColors::GRID
        ));
        svg.push('\n');

        // Grid lines and axis labels
        if config.show_grid {
            svg.push_str(&self.render_grid(
                margin,
                plot_width,
                plot_height,
                x_min,
                x_max,
                y_min,
                y_max,
                config,
            ));
        }

        // Target line (30x)
        if config.show_target_line {
            let target_y = self.scale_y(
                TARGET_OVERHEAD_MULTIPLE,
                y_min,
                y_max,
                plot_height,
                config.log_y,
            );
            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2" stroke-dasharray="8,4"/>"#,
                margin, 50.0 + target_y, margin + plot_width, 50.0 + target_y, ChartColors::TARGET_LINE
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">30x target</text>"#,
                margin + plot_width + 5.0, 50.0 + target_y + 4.0, ChartColors::TARGET_LINE
            ));
            svg.push('\n');
        }

        // Alert line (35x)
        if config.show_alert_line {
            let alert_y = self.scale_y(
                REGRESSION_ALERT_THRESHOLD,
                y_min,
                y_max,
                plot_height,
                config.log_y,
            );
            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2" stroke-dasharray="4,4"/>"#,
                margin, 50.0 + alert_y, margin + plot_width, 50.0 + alert_y, ChartColors::ALERT_LINE
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">35x alert</text>"#,
                margin + plot_width + 5.0, 50.0 + alert_y + 4.0, ChartColors::ALERT_LINE
            ));
            svg.push('\n');
        }

        // Data points by category
        let mut categories: BTreeMap<String, Vec<&ChartDataPoint>> = BTreeMap::new();
        for point in data {
            categories
                .entry(point.category.clone())
                .or_default()
                .push(point);
        }

        for (category, points) in &categories {
            let color = match category.as_str() {
                "Native" => ChartColors::NATIVE,
                "GKR" => ChartColors::GKR,
                "Halo2" => ChartColors::HALO2,
                _ => ChartColors::GKR,
            };

            // Draw line connecting points
            if points.len() > 1 {
                let mut path = String::from("M");
                let mut sorted_points: Vec<_> = points.iter().collect();
                sorted_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

                for (i, point) in sorted_points.iter().enumerate() {
                    let x = self.scale_x(point.x, x_min, x_max, plot_width, config.log_x);
                    let y = self.scale_y(point.y, y_min, y_max, plot_height, config.log_y);
                    if i == 0 {
                        path.push_str(&format!(" {},{}", margin + x, 50.0 + y));
                    } else {
                        path.push_str(&format!(" L {},{}", margin + x, 50.0 + y));
                    }
                }
                svg.push_str(&format!(
                    r#"  <path d="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
                    path, color
                ));
                svg.push('\n');
            }

            // Draw data points
            for point in points {
                let x = self.scale_x(point.x, x_min, x_max, plot_width, config.log_x);
                let y = self.scale_y(point.y, y_min, y_max, plot_height, config.log_y);
                svg.push_str(&format!(
                    r#"  <circle cx="{}" cy="{}" r="5" fill="{}" stroke="white" stroke-width="1.5"/>"#,
                    margin + x, 50.0 + y, color
                ));
                svg.push('\n');
            }
        }

        // X-axis label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}">{}</text>"#,
            margin + plot_width / 2.0,
            config.height as f64 - 10.0,
            ChartColors::TEXT,
            config.x_label
        ));
        svg.push('\n');

        // Y-axis label (rotated)
        svg.push_str(&format!(
            r#"  <text x="15" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}" transform="rotate(-90, 15, {})">{}</text>"#,
            50.0 + plot_height / 2.0,
            ChartColors::TEXT,
            50.0 + plot_height / 2.0,
            config.y_label
        ));
        svg.push('\n');

        // Legend
        if config.show_legend {
            svg.push_str(
                &self.render_legend(&categories.keys().cloned().collect::<Vec<_>>(), config),
            );
        }

        svg.push_str("</svg>\n");
        svg
    }

    /// Calculates the data ranges for axis scaling.
    fn calculate_ranges(
        &self,
        data: &[ChartDataPoint],
        config: &ChartConfig,
    ) -> (f64, f64, f64, f64) {
        if data.is_empty() {
            return (1.0, 100.0, 0.0, 50.0);
        }

        let x_values: Vec<f64> = data.iter().map(|p| p.x).collect();
        let y_values: Vec<f64> = data.iter().map(|p| p.y).collect();

        let x_min = x_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = 0.0; // Always start Y at 0
        let mut y_max = y_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Ensure Y max includes target and alert lines if shown
        if config.show_target_line {
            y_max = y_max.max(TARGET_OVERHEAD_MULTIPLE * 1.2);
        }
        if config.show_alert_line {
            y_max = y_max.max(REGRESSION_ALERT_THRESHOLD * 1.2);
        }

        // Add some padding
        let x_padding = (x_max - x_min) * 0.1;
        let y_padding = y_max * 0.1;

        (
            x_min - x_padding,
            x_max + x_padding,
            y_min,
            y_max + y_padding,
        )
    }

    /// Scales X value to plot coordinates.
    fn scale_x(&self, value: f64, min: f64, max: f64, width: f64, log_scale: bool) -> f64 {
        if log_scale && value > 0.0 && min > 0.0 {
            let log_val = value.log10();
            let log_min = min.log10();
            let log_max = max.log10();
            (log_val - log_min) / (log_max - log_min) * width
        } else {
            (value - min) / (max - min) * width
        }
    }

    /// Scales Y value to plot coordinates (inverted for SVG).
    fn scale_y(&self, value: f64, min: f64, max: f64, height: f64, log_scale: bool) -> f64 {
        if log_scale && value > 0.0 && min > 0.0 {
            let log_val = value.log10();
            let log_min = min.max(0.1).log10();
            let log_max = max.log10();
            height - (log_val - log_min) / (log_max - log_min) * height
        } else {
            height - (value - min) / (max - min) * height
        }
    }

    /// Renders grid lines.
    fn render_grid(
        &self,
        margin: f64,
        width: f64,
        height: f64,
        _x_min: f64,
        _x_max: f64,
        _y_min: f64,
        y_max: f64,
        config: &ChartConfig,
    ) -> String {
        let mut svg = String::new();

        // Horizontal grid lines
        let y_ticks = 5;
        for i in 0..=y_ticks {
            let y_val = (i as f64 / y_ticks as f64) * y_max;
            let y_pos = height - (i as f64 / y_ticks as f64) * height;

            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1" opacity="0.3"/>"#,
                margin, 50.0 + y_pos, margin + width, 50.0 + y_pos, ChartColors::GRID
            ));
            svg.push('\n');

            // Y-axis tick label
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="{}">{:.0}</text>"#,
                margin - 5.0, 50.0 + y_pos + 4.0, ChartColors::TEXT, y_val
            ));
            svg.push('\n');
        }

        svg
    }

    /// Renders a legend.
    fn render_legend(&self, categories: &[String], config: &ChartConfig) -> String {
        let mut svg = String::new();
        let legend_x = config.width as f64 - 120.0;
        let legend_y = 70.0;

        svg.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="110" height="{}" fill="white" stroke="{}" stroke-width="1" rx="3"/>"#,
            legend_x - 5.0, legend_y - 15.0, 20 * categories.len() + 10, ChartColors::GRID
        ));
        svg.push('\n');

        for (i, category) in categories.iter().enumerate() {
            let color = match category.as_str() {
                "Native" => ChartColors::NATIVE,
                "GKR" => ChartColors::GKR,
                "Halo2" => ChartColors::HALO2,
                _ => ChartColors::GKR,
            };
            let y = legend_y + i as f64 * 20.0;

            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="15" height="15" fill="{}" rx="2"/>"#,
                legend_x,
                y - 10.0,
                color
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
                legend_x + 20.0, y + 2.0, ChartColors::TEXT, category
            ));
            svg.push('\n');
        }

        svg
    }

    /// Generates an ASCII table for terminal/markdown output.
    pub fn generate_ascii_table(&self, report: &OverheadReport) -> String {
        let mut table = String::new();

        // Header
        table.push_str("\n");
        table.push_str(&format!("{:─^78}\n", " HELIX Overhead Validation Results "));
        table.push_str(&format!(
            "│{:^15}│{:^12}│{:^12}│{:^12}│{:^12}│{:^10}│\n",
            "Model", "Params", "Native(μs)", "Proof(μs)", "Overhead", "Status"
        ));
        table.push_str(&format!(
            "│{:─^15}│{:─^12}│{:─^12}│{:─^12}│{:─^12}│{:─^10}│\n",
            "", "", "", "", "", ""
        ));

        // Data rows
        for (name, data) in &report.by_model_size {
            if let Some(ref gkr) = data.gkr_overhead {
                let native_us = gkr.native_time.as_secs_f64() * 1_000_000.0;
                let proof_us = gkr.proof_time.as_secs_f64() * 1_000_000.0;
                let status = if gkr.meets_target {
                    "✓ PASS"
                } else if gkr.overhead <= REGRESSION_ALERT_THRESHOLD {
                    "! WARN"
                } else {
                    "✗ FAIL"
                };

                table.push_str(&format!(
                    "│{:^15}│{:^12}│{:^12.1}│{:^12.1}│{:^11.1}x│{:^10}│\n",
                    name, data.param_count, native_us, proof_us, gkr.overhead, status
                ));
            }
        }

        // Footer with summary
        table.push_str(&format!(
            "│{:─^15}│{:─^12}│{:─^12}│{:─^12}│{:─^12}│{:─^10}│\n",
            "", "", "", "", "", ""
        ));
        table.push_str(&format!(
            "│ Target: ≤{:.0}x │ Alert: ≤{:.0}x │{:^37}│\n",
            TARGET_OVERHEAD_MULTIPLE,
            REGRESSION_ALERT_THRESHOLD,
            if report.summary.passed {
                "ALL TESTS PASSED"
            } else {
                "TESTS FAILED"
            }
        ));
        table.push_str(&format!("{:─^78}\n", ""));

        table
    }

    /// Generates all charts and tables for the report.
    pub fn generate_all(
        &self,
        report: &OverheadReport,
        scaling: Option<&ScalingAnalysis>,
    ) -> std::io::Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        // Overhead chart
        paths.push(self.generate_overhead_svg(report, "overhead_chart.svg")?);

        // Timing chart
        paths.push(self.generate_timing_svg(report, "timing_chart.svg")?);

        // Scaling chart (if analysis provided)
        if let Some(analysis) = scaling {
            paths.push(self.generate_scaling_svg(analysis, "scaling_chart.svg")?);
        }

        // ASCII table
        let table = self.generate_ascii_table(report);
        let table_path = self.output_dir.join("results_table.txt");
        std::fs::write(&table_path, &table)?;
        paths.push(table_path);

        Ok(paths)
    }
}

impl Default for ChartGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates a markdown summary with embedded chart references.
pub fn generate_markdown_with_charts(report: &OverheadReport, chart_dir: &str) -> String {
    let mut md = String::new();

    md.push_str("# HELIX Overhead Validation Report\n\n");
    md.push_str(&format!("**Generated:** {}\n\n", report.timestamp));

    if let Some(ref commit) = report.git_commit {
        md.push_str(&format!("**Commit:** `{}`\n\n", commit));
    }

    md.push_str("## Summary\n\n");
    md.push_str(&format!(
        "- **Target Overhead:** ≤{:.0}x\n",
        TARGET_OVERHEAD_MULTIPLE
    ));
    md.push_str(&format!(
        "- **Alert Threshold:** ≤{:.0}x\n",
        REGRESSION_ALERT_THRESHOLD
    ));
    md.push_str(&format!(
        "- **Average Overhead:** {:.1}x\n",
        report.summary.avg_overhead
    ));
    md.push_str(&format!(
        "- **Status:** {}\n\n",
        if report.summary.passed {
            "✅ PASSED"
        } else {
            "❌ FAILED"
        }
    ));

    md.push_str("## Overhead Analysis\n\n");
    md.push_str(&format!(
        "![Overhead Chart]({}/overhead_chart.svg)\n\n",
        chart_dir
    ));

    md.push_str("## Timing Analysis\n\n");
    md.push_str(&format!(
        "![Timing Chart]({}/timing_chart.svg)\n\n",
        chart_dir
    ));

    md.push_str("## Detailed Results\n\n");
    md.push_str("| Model | Parameters | Native (μs) | Proof (μs) | Overhead | Status |\n");
    md.push_str("|-------|------------|-------------|------------|----------|--------|\n");

    for (name, data) in &report.by_model_size {
        if let Some(ref gkr) = data.gkr_overhead {
            let native_us = gkr.native_time.as_secs_f64() * 1_000_000.0;
            let proof_us = gkr.proof_time.as_secs_f64() * 1_000_000.0;
            let status = if gkr.meets_target {
                "✅"
            } else if gkr.overhead <= REGRESSION_ALERT_THRESHOLD {
                "⚠️"
            } else {
                "❌"
            };

            md.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.1}x | {} |\n",
                name, data.param_count, native_us, proof_us, gkr.overhead, status
            ));
        }
    }

    md.push_str("\n## Success Criteria\n\n");
    md.push_str(
        "- **≤30x overhead:** All measurements must achieve overhead ≤30x native computation\n",
    );
    md.push_str("- **Regression alert:** CI alerts if any measurement exceeds 35x overhead\n");
    md.push_str("- **CI integration:** Automatic regression detection with baseline comparison\n");

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_generator_creation() {
        let gen = ChartGenerator::new();
        assert!(gen.output_dir.to_string_lossy().contains("charts"));
    }

    #[test]
    fn test_chart_data_point() {
        let point = ChartDataPoint {
            x: 10000.0,
            y: 15.5,
            label: "10k_params".to_string(),
            category: "GKR".to_string(),
        };
        assert_eq!(point.x, 10000.0);
        assert_eq!(point.y, 15.5);
    }

    #[test]
    fn test_scaling() {
        let gen = ChartGenerator::new();
        let config = ChartConfig::default();

        // Test linear scaling
        let scaled = gen.scale_x(50.0, 0.0, 100.0, 100.0, false);
        assert!((scaled - 50.0).abs() < 0.1);

        // Test Y scaling (inverted)
        let scaled_y = gen.scale_y(50.0, 0.0, 100.0, 100.0, false);
        assert!((scaled_y - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_ascii_table_generation() {
        let report = OverheadReport::new("Test Report");
        let gen = ChartGenerator::new();
        let table = gen.generate_ascii_table(&report);
        assert!(table.contains("Overhead"));
        assert!(table.contains("Target"));
    }
}
