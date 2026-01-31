//! Automated Optimization Report Generator.
//!
//! Generates comprehensive reports analyzing circuit performance and
//! providing actionable optimization recommendations.
//!
//! # Report Types
//!
//! - **Summary Report**: High-level overview suitable for quick review
//! - **Detailed Report**: Full analysis with all metrics and recommendations
//! - **Comparison Report**: Side-by-side comparison of implementations
//! - **Trend Report**: Historical performance analysis
//!
//! # Usage
//!
//! ```ignore
//! let generator = ReportGenerator::new();
//! let report = generator.generate(&profile, ReportFormat::Detailed);
//! println!("{}", report.render());
//! ```

use super::{
    CircuitProfile, ConstraintProfile, TimingProfile, Bottleneck, BottleneckType,
    constraint_counter::{OperationCost, OptimizationTechnique},
};
use std::time::Duration;
use std::collections::HashMap;

/// Format for the optimization report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Brief summary with key metrics.
    Summary,
    /// Full detailed analysis.
    Detailed,
    /// Markdown format for documentation.
    Markdown,
    /// JSON format for programmatic consumption.
    Json,
    /// HTML format for web display.
    Html,
}

/// Priority level for an optimization recommendation.
/// Ordered from lowest (Info) to highest (Critical) for comparison purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    /// Info - informational only.
    Info,
    /// Low - minor improvement.
    Low,
    /// Medium - noticeable improvement.
    Medium,
    /// High - significant impact on performance.
    High,
    /// Critical - must fix to meet targets.
    Critical,
}

impl std::fmt::Display for OptimizationPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "🔴 CRITICAL"),
            Self::High => write!(f, "🟠 HIGH"),
            Self::Medium => write!(f, "🟡 MEDIUM"),
            Self::Low => write!(f, "🟢 LOW"),
            Self::Info => write!(f, "ℹ️ INFO"),
        }
    }
}

/// A single optimization recommendation.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Priority level.
    pub priority: OptimizationPriority,
    /// Title of the recommendation.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Affected component/operation.
    pub affected_component: String,
    /// Current performance.
    pub current_metric: String,
    /// Expected performance after optimization.
    pub expected_metric: String,
    /// Implementation steps.
    pub implementation_steps: Vec<String>,
    /// Estimated effort (1-5).
    pub effort_estimate: u8,
    /// Estimated impact (1-5).
    pub impact_estimate: u8,
    /// Related code locations.
    pub code_locations: Vec<String>,
}

impl OptimizationRecommendation {
    /// Creates a new recommendation.
    pub fn new(priority: OptimizationPriority, title: &str, description: &str) -> Self {
        Self {
            priority,
            title: title.to_string(),
            description: description.to_string(),
            affected_component: String::new(),
            current_metric: String::new(),
            expected_metric: String::new(),
            implementation_steps: Vec::new(),
            effort_estimate: 3,
            impact_estimate: 3,
            code_locations: Vec::new(),
        }
    }

    /// Sets the affected component.
    pub fn with_component(mut self, component: &str) -> Self {
        self.affected_component = component.to_string();
        self
    }

    /// Sets the current and expected metrics.
    pub fn with_metrics(mut self, current: &str, expected: &str) -> Self {
        self.current_metric = current.to_string();
        self.expected_metric = expected.to_string();
        self
    }

    /// Sets implementation steps.
    pub fn with_steps(mut self, steps: Vec<&str>) -> Self {
        self.implementation_steps = steps.into_iter().map(String::from).collect();
        self
    }

    /// Sets effort and impact estimates.
    pub fn with_estimates(mut self, effort: u8, impact: u8) -> Self {
        self.effort_estimate = effort.min(5);
        self.impact_estimate = impact.min(5);
        self
    }

    /// Adds code locations.
    pub fn with_locations(mut self, locations: Vec<&str>) -> Self {
        self.code_locations = locations.into_iter().map(String::from).collect();
        self
    }

    /// Returns the ROI score (impact / effort).
    pub fn roi_score(&self) -> f64 {
        self.impact_estimate as f64 / self.effort_estimate.max(1) as f64
    }
}

/// Analysis of constraint hotspots.
#[derive(Debug, Clone)]
pub struct ConstraintHotspot {
    /// Name of the operation.
    pub operation: String,
    /// Number of constraints.
    pub constraint_count: usize,
    /// Percentage of total.
    pub percentage: f64,
    /// Optimization available.
    pub optimization: OptimizationTechnique,
    /// Potential savings.
    pub potential_savings: usize,
}

/// Analysis of performance bottlenecks.
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// List of identified bottlenecks.
    pub bottlenecks: Vec<AnalyzedBottleneck>,
    /// Overall bottleneck score (0-100, lower is better).
    pub score: u32,
    /// Primary bottleneck category.
    pub primary_category: BottleneckCategory,
    /// Estimated total savings if all fixed.
    pub total_potential_savings: PotentialSavings,
}

/// A fully analyzed bottleneck.
#[derive(Debug, Clone)]
pub struct AnalyzedBottleneck {
    /// Original bottleneck.
    pub bottleneck: Bottleneck,
    /// Root cause analysis.
    pub root_cause: String,
    /// Recommended fix.
    pub recommended_fix: String,
    /// Dependencies on other fixes.
    pub dependencies: Vec<String>,
    /// Risk of the fix.
    pub fix_risk: FixRisk,
}

/// Risk level of a fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixRisk {
    /// No risk.
    None,
    /// Low risk.
    Low,
    /// Medium risk.
    Medium,
    /// High risk.
    High,
}

/// Category of bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckCategory {
    /// Constraint-related.
    Constraints,
    /// Timing-related.
    Timing,
    /// Memory-related.
    Memory,
    /// Architecture-related.
    Architecture,
}

/// Potential savings from optimizations.
#[derive(Debug, Clone, Default)]
pub struct PotentialSavings {
    /// Constraint reduction.
    pub constraint_reduction: usize,
    /// Time reduction.
    pub time_reduction: Duration,
    /// Memory reduction.
    pub memory_reduction: usize,
    /// Proof size reduction.
    pub proof_size_reduction: usize,
}

/// Complete optimization report.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Report title.
    pub title: String,
    /// Report timestamp.
    pub timestamp: std::time::SystemTime,
    /// Executive summary.
    pub summary: String,
    /// Overall score (0-100).
    pub score: u32,
    /// Whether targets are met.
    pub meets_targets: bool,
    /// Key metrics.
    pub metrics: ReportMetrics,
    /// Recommendations ordered by priority.
    pub recommendations: Vec<OptimizationRecommendation>,
    /// Constraint hotspots.
    pub hotspots: Vec<ConstraintHotspot>,
    /// Bottleneck analysis.
    pub bottleneck_analysis: BottleneckAnalysis,
    /// Detailed sections.
    pub sections: Vec<ReportSection>,
}

/// Key metrics in the report.
#[derive(Debug, Clone, Default)]
pub struct ReportMetrics {
    /// Total constraints.
    pub total_constraints: usize,
    /// Constraint target.
    pub constraint_target: usize,
    /// Total time.
    pub total_time: Duration,
    /// Time target.
    pub time_target: Duration,
    /// Memory usage.
    pub memory_usage: usize,
    /// Memory target.
    pub memory_target: usize,
    /// K parameter.
    pub k_parameter: u32,
    /// Proof size.
    pub proof_size: usize,
}

/// A section of the report.
#[derive(Debug, Clone)]
pub struct ReportSection {
    /// Section title.
    pub title: String,
    /// Section content.
    pub content: String,
    /// Subsections.
    pub subsections: Vec<ReportSubsection>,
}

/// A subsection of the report.
#[derive(Debug, Clone)]
pub struct ReportSubsection {
    /// Subsection title.
    pub title: String,
    /// Subsection content.
    pub content: String,
}

/// Report generator.
pub struct ReportGenerator {
    /// Configuration.
    config: ReportConfig,
}

/// Configuration for report generation.
#[derive(Debug, Clone)]
pub struct ReportConfig {
    /// Include implementation steps.
    pub include_implementation: bool,
    /// Include code locations.
    pub include_code_locations: bool,
    /// Maximum recommendations to include.
    pub max_recommendations: usize,
    /// Minimum priority to include.
    pub min_priority: OptimizationPriority,
    /// Target timing (500ms for HELIX).
    pub target_time: Duration,
    /// Target constraint count.
    pub target_constraints: usize,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            include_implementation: true,
            include_code_locations: true,
            max_recommendations: 20,
            min_priority: OptimizationPriority::Low,
            target_time: Duration::from_millis(500),
            target_constraints: 500_000,
        }
    }
}

impl ReportGenerator {
    /// Creates a new report generator.
    pub fn new() -> Self {
        Self {
            config: ReportConfig::default(),
        }
    }

    /// Creates a report generator with custom configuration.
    pub fn with_config(config: ReportConfig) -> Self {
        Self { config }
    }

    /// Generates a report from a circuit profile.
    pub fn generate(&self, profile: &CircuitProfile, format: ReportFormat) -> OptimizationReport {
        let recommendations = self.generate_recommendations(profile);
        let hotspots = self.identify_hotspots(profile);
        let bottleneck_analysis = self.analyze_bottlenecks(profile);

        let score = self.calculate_score(profile, &recommendations);
        let summary = self.generate_summary(profile, &recommendations, score);

        let sections = match format {
            ReportFormat::Summary => self.generate_summary_sections(profile),
            ReportFormat::Detailed => self.generate_detailed_sections(profile),
            ReportFormat::Markdown => self.generate_detailed_sections(profile),
            ReportFormat::Json => self.generate_detailed_sections(profile),
            ReportFormat::Html => self.generate_detailed_sections(profile),
        };

        OptimizationReport {
            title: format!("Circuit Optimization Report: {}", profile.name),
            timestamp: std::time::SystemTime::now(),
            summary,
            score,
            meets_targets: profile.meets_targets,
            metrics: ReportMetrics {
                total_constraints: profile.constraints.total_constraints,
                constraint_target: self.config.target_constraints,
                total_time: profile.timing.total_time,
                time_target: self.config.target_time,
                memory_usage: profile.memory_bytes,
                memory_target: 100_000_000, // 100MB
                k_parameter: profile.k,
                proof_size: 0, // Would need proof generation for this
            },
            recommendations,
            hotspots,
            bottleneck_analysis,
            sections,
        }
    }

    /// Generates recommendations from a circuit profile.
    fn generate_recommendations(&self, profile: &CircuitProfile) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check timing target
        if profile.timing.total_time > self.config.target_time {
            let excess = profile.timing.total_time - self.config.target_time;
            let priority = if excess > Duration::from_secs(1) {
                OptimizationPriority::Critical
            } else if excess > Duration::from_millis(500) {
                OptimizationPriority::High
            } else {
                OptimizationPriority::Medium
            };

            recommendations.push(
                OptimizationRecommendation::new(
                    priority,
                    "Reduce Proof Generation Time",
                    &format!(
                        "Current time ({:?}) exceeds target ({:?}) by {:?}",
                        profile.timing.total_time, self.config.target_time, excess
                    ),
                )
                .with_component("overall")
                .with_metrics(
                    &format!("{:?}", profile.timing.total_time),
                    &format!("{:?}", self.config.target_time),
                )
                .with_steps(vec![
                    "Profile constraint hotspots",
                    "Apply Freivalds verification to matrix operations",
                    "Optimize lookup table sizes",
                    "Parallelize witness generation",
                ])
                .with_estimates(4, 5)
            );
        }

        // Check constraint target
        if profile.constraints.total_constraints > self.config.target_constraints {
            let excess = profile.constraints.total_constraints - self.config.target_constraints;
            let percentage = (excess as f64 / self.config.target_constraints as f64) * 100.0;

            recommendations.push(
                OptimizationRecommendation::new(
                    if percentage > 100.0 { OptimizationPriority::Critical }
                    else if percentage > 50.0 { OptimizationPriority::High }
                    else { OptimizationPriority::Medium },
                    "Reduce Constraint Count",
                    &format!(
                        "Current constraints ({}) exceed target ({}) by {} ({:.1}%)",
                        profile.constraints.total_constraints,
                        self.config.target_constraints,
                        excess,
                        percentage
                    ),
                )
                .with_component("constraints")
                .with_estimates(4, 5)
            );
        }

        // Analyze bottlenecks for recommendations
        for bottleneck in &profile.bottlenecks {
            let priority = match bottleneck.impact {
                80..=100 => OptimizationPriority::Critical,
                60..=79 => OptimizationPriority::High,
                40..=59 => OptimizationPriority::Medium,
                20..=39 => OptimizationPriority::Low,
                _ => OptimizationPriority::Info,
            };

            if priority >= self.config.min_priority {
                recommendations.push(
                    OptimizationRecommendation::new(
                        priority,
                        &format!("Fix: {:?}", bottleneck.bottleneck_type),
                        &bottleneck.description,
                    )
                    .with_component(&format!("{:?}", bottleneck.bottleneck_type))
                    .with_steps(vec![&bottleneck.suggestion])
                    .with_estimates(3, (bottleneck.impact / 20).min(5) as u8)
                );
            }
        }

        // Specific optimization recommendations
        if profile.constraints.breakdown.lookup_constraints > profile.constraints.total_constraints / 3 {
            recommendations.push(
                OptimizationRecommendation::new(
                    OptimizationPriority::Medium,
                    "Optimize Lookup Table Size",
                    "Lookup constraints are a significant portion of total. Consider range compression.",
                )
                .with_component("lookup")
                .with_steps(vec![
                    "Analyze actual lookup value distribution",
                    "Implement range compression for sparse lookups",
                    "Consider combined multi-value lookups",
                ])
                .with_estimates(3, 3)
            );
        }

        // Witness generation optimization
        if profile.timing.witness_generation > profile.timing.total_time / 2 {
            recommendations.push(
                OptimizationRecommendation::new(
                    OptimizationPriority::High,
                    "Parallelize Witness Generation",
                    "Witness generation is the dominant time cost. Parallelization can help.",
                )
                .with_component("witness")
                .with_steps(vec![
                    "Use rayon for parallel iteration over weight matrices",
                    "Cache intermediate computations",
                    "Batch hash computations",
                ])
                .with_estimates(2, 4)
            );
        }

        // Sort by priority and ROI
        recommendations.sort_by(|a, b| {
            match b.priority.cmp(&a.priority) {
                std::cmp::Ordering::Equal => {
                    b.roi_score().partial_cmp(&a.roi_score()).unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            }
        });

        // Limit to max recommendations
        recommendations.truncate(self.config.max_recommendations);

        recommendations
    }

    /// Identifies constraint hotspots.
    fn identify_hotspots(&self, profile: &CircuitProfile) -> Vec<ConstraintHotspot> {
        profile.constraints.operation_costs.iter()
            .filter(|op| op.percentage > 5.0)
            .map(|op| ConstraintHotspot {
                operation: op.name.clone(),
                constraint_count: op.constraint_count,
                percentage: op.percentage,
                optimization: op.optimization_potential.technique,
                potential_savings: op.optimization_potential.constraint_reduction,
            })
            .collect()
    }

    /// Analyzes bottlenecks.
    fn analyze_bottlenecks(&self, profile: &CircuitProfile) -> BottleneckAnalysis {
        let analyzed: Vec<AnalyzedBottleneck> = profile.bottlenecks.iter()
            .map(|b| AnalyzedBottleneck {
                bottleneck: b.clone(),
                root_cause: self.analyze_root_cause(b),
                recommended_fix: b.suggestion.clone(),
                dependencies: Vec::new(),
                fix_risk: match b.bottleneck_type {
                    BottleneckType::HighConstraintCount => FixRisk::Low,
                    BottleneckType::LookupInefficiency => FixRisk::Low,
                    BottleneckType::SlowWitnessGeneration => FixRisk::Low,
                    _ => FixRisk::Medium,
                },
            })
            .collect();

        let primary_category = if !profile.bottlenecks.is_empty() {
            match profile.bottlenecks[0].bottleneck_type {
                BottleneckType::HighConstraintCount => BottleneckCategory::Constraints,
                BottleneckType::SlowWitnessGeneration => BottleneckCategory::Timing,
                BottleneckType::HighMemoryUsage => BottleneckCategory::Memory,
                _ => BottleneckCategory::Architecture,
            }
        } else {
            BottleneckCategory::Constraints
        };

        let total_constraint_savings: usize = profile.bottlenecks.iter()
            .map(|b| b.estimated_savings.constraint_reduction)
            .sum();
        let total_time_savings: Duration = profile.bottlenecks.iter()
            .map(|b| b.estimated_savings.time_reduction)
            .sum();

        BottleneckAnalysis {
            bottlenecks: analyzed,
            score: 100 - profile.performance_score,
            primary_category,
            total_potential_savings: PotentialSavings {
                constraint_reduction: total_constraint_savings,
                time_reduction: total_time_savings,
                memory_reduction: 0,
                proof_size_reduction: 0,
            },
        }
    }

    /// Analyzes root cause of a bottleneck.
    fn analyze_root_cause(&self, bottleneck: &Bottleneck) -> String {
        match bottleneck.bottleneck_type {
            BottleneckType::HighConstraintCount => {
                "Operation complexity exceeds optimal constraint budget".to_string()
            }
            BottleneckType::LookupInefficiency => {
                "Lookup table size or usage pattern is suboptimal".to_string()
            }
            BottleneckType::SlowWitnessGeneration => {
                "Sequential witness computation creates bottleneck".to_string()
            }
            BottleneckType::HighMemoryUsage => {
                "Large intermediate values or unoptimized data structures".to_string()
            }
            BottleneckType::GateInefficiency => {
                "Gate selection not optimized for circuit constraints".to_string()
            }
            BottleneckType::ExcessiveCopyConstraints => {
                "Too many equality constraints between cells".to_string()
            }
            BottleneckType::LargePublicInputs => {
                "Public input count creates verification overhead".to_string()
            }
        }
    }

    /// Calculates overall score.
    fn calculate_score(&self, profile: &CircuitProfile, recommendations: &[OptimizationRecommendation]) -> u32 {
        let mut score = profile.performance_score;

        // Penalize for critical recommendations
        let critical_count = recommendations.iter()
            .filter(|r| r.priority == OptimizationPriority::Critical)
            .count();
        score = score.saturating_sub((critical_count * 10) as u32);

        // Penalize for not meeting targets
        if !profile.meets_targets {
            score = score.saturating_sub(20);
        }

        score
    }

    /// Generates executive summary.
    fn generate_summary(
        &self,
        profile: &CircuitProfile,
        recommendations: &[OptimizationRecommendation],
        score: u32,
    ) -> String {
        let status = if profile.meets_targets {
            "✅ Meets performance targets"
        } else {
            "⚠️ Does not meet performance targets"
        };

        let critical_count = recommendations.iter()
            .filter(|r| r.priority == OptimizationPriority::Critical)
            .count();

        format!(
            "{}\n\n\
             Score: {}/100\n\
             Total Constraints: {} (target: {})\n\
             Total Time: {:?} (target: {:?})\n\n\
             {} critical issues, {} total recommendations\n\n\
             Top recommendation: {}",
            status,
            score,
            profile.constraints.total_constraints,
            self.config.target_constraints,
            profile.timing.total_time,
            self.config.target_time,
            critical_count,
            recommendations.len(),
            recommendations.first()
                .map(|r| r.title.as_str())
                .unwrap_or("No recommendations")
        )
    }

    /// Generates summary sections.
    fn generate_summary_sections(&self, profile: &CircuitProfile) -> Vec<ReportSection> {
        vec![
            ReportSection {
                title: "Performance Summary".to_string(),
                content: format!(
                    "Constraints: {}\nTime: {:?}\nK: {}",
                    profile.constraints.total_constraints,
                    profile.timing.total_time,
                    profile.k
                ),
                subsections: vec![],
            },
        ]
    }

    /// Generates detailed sections.
    fn generate_detailed_sections(&self, profile: &CircuitProfile) -> Vec<ReportSection> {
        vec![
            ReportSection {
                title: "Constraint Analysis".to_string(),
                content: format!(
                    "Total: {}\nGates: {}\nLookups: {}\nCopy: {}",
                    profile.constraints.total_constraints,
                    profile.constraints.breakdown.gate_constraints,
                    profile.constraints.breakdown.lookup_constraints,
                    profile.constraints.breakdown.copy_constraints,
                ),
                subsections: vec![
                    ReportSubsection {
                        title: "Gate Distribution".to_string(),
                        content: format!(
                            "Gates comprise {:.1}% of total constraints",
                            profile.constraints.breakdown.gate_constraints as f64
                                / profile.constraints.total_constraints.max(1) as f64 * 100.0
                        ),
                    },
                ],
            },
            ReportSection {
                title: "Timing Analysis".to_string(),
                content: format!(
                    "Total: {:?}\nWitness: {:?}\nSynthesis: {:?}\nVerify: {:?}",
                    profile.timing.total_time,
                    profile.timing.witness_generation,
                    profile.timing.synthesis,
                    profile.timing.verification,
                ),
                subsections: vec![],
            },
            ReportSection {
                title: "Bottleneck Analysis".to_string(),
                content: format!("{} bottlenecks identified", profile.bottlenecks.len()),
                subsections: profile.bottlenecks.iter().map(|b| {
                    ReportSubsection {
                        title: format!("{:?}", b.bottleneck_type),
                        content: format!(
                            "{}\nImpact: {}\nSuggestion: {}",
                            b.description, b.impact, b.suggestion
                        ),
                    }
                }).collect(),
            },
        ]
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationReport {
    /// Renders the report as a string.
    pub fn render(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("╔═══════════════════════════════════════════════════════════════╗\n"));
        output.push_str(&format!("║ {:^61} ║\n", self.title));
        output.push_str(&format!("╚═══════════════════════════════════════════════════════════════╝\n\n"));

        output.push_str("📊 EXECUTIVE SUMMARY\n");
        output.push_str("═══════════════════════════════════════════════════════════════════\n");
        output.push_str(&self.summary);
        output.push_str("\n\n");

        output.push_str("📈 KEY METRICS\n");
        output.push_str("═══════════════════════════════════════════════════════════════════\n");
        output.push_str(&format!(
            "Constraints: {} / {} ({:.1}% of target)\n",
            self.metrics.total_constraints,
            self.metrics.constraint_target,
            self.metrics.total_constraints as f64 / self.metrics.constraint_target as f64 * 100.0
        ));
        output.push_str(&format!(
            "Time: {:?} / {:?} ({:.1}% of target)\n",
            self.metrics.total_time,
            self.metrics.time_target,
            self.metrics.total_time.as_nanos() as f64 / self.metrics.time_target.as_nanos() as f64 * 100.0
        ));
        output.push_str(&format!("K Parameter: {}\n", self.metrics.k_parameter));
        output.push_str(&format!("Memory: {:.2} MB\n", self.metrics.memory_usage as f64 / 1_000_000.0));
        output.push_str("\n");

        if !self.recommendations.is_empty() {
            output.push_str("💡 RECOMMENDATIONS\n");
            output.push_str("═══════════════════════════════════════════════════════════════════\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                output.push_str(&format!(
                    "{}. [{}] {}\n   {}\n   Effort: {}/5 | Impact: {}/5 | ROI: {:.2}\n\n",
                    i + 1,
                    rec.priority,
                    rec.title,
                    rec.description,
                    rec.effort_estimate,
                    rec.impact_estimate,
                    rec.roi_score()
                ));
            }
        }

        if !self.hotspots.is_empty() {
            output.push_str("🔥 CONSTRAINT HOTSPOTS\n");
            output.push_str("═══════════════════════════════════════════════════════════════════\n");
            for hotspot in &self.hotspots {
                output.push_str(&format!(
                    "• {}: {} constraints ({:.1}%) - {:?}\n",
                    hotspot.operation,
                    hotspot.constraint_count,
                    hotspot.percentage,
                    hotspot.optimization
                ));
            }
            output.push_str("\n");
        }

        output.push_str(&format!(
            "═══════════════════════════════════════════════════════════════════\n\
             Score: {}/100 | Status: {}\n",
            self.score,
            if self.meets_targets { "✅ PASS" } else { "❌ FAIL" }
        ));

        output
    }

    /// Renders the report as Markdown.
    pub fn render_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("# {}\n\n", self.title));
        output.push_str("## Executive Summary\n\n");
        output.push_str(&self.summary);
        output.push_str("\n\n");

        output.push_str("## Key Metrics\n\n");
        output.push_str("| Metric | Current | Target | Status |\n");
        output.push_str("|--------|---------|--------|--------|\n");
        output.push_str(&format!(
            "| Constraints | {} | {} | {} |\n",
            self.metrics.total_constraints,
            self.metrics.constraint_target,
            if self.metrics.total_constraints <= self.metrics.constraint_target { "✅" } else { "❌" }
        ));
        output.push_str(&format!(
            "| Time | {:?} | {:?} | {} |\n",
            self.metrics.total_time,
            self.metrics.time_target,
            if self.metrics.total_time <= self.metrics.time_target { "✅" } else { "❌" }
        ));
        output.push_str("\n");

        if !self.recommendations.is_empty() {
            output.push_str("## Recommendations\n\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                output.push_str(&format!(
                    "### {}. {} ({})\n\n{}\n\n",
                    i + 1, rec.title, rec.priority, rec.description
                ));

                if !rec.implementation_steps.is_empty() {
                    output.push_str("**Implementation Steps:**\n\n");
                    for step in &rec.implementation_steps {
                        output.push_str(&format!("1. {}\n", step));
                    }
                    output.push_str("\n");
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_builder() {
        let rec = OptimizationRecommendation::new(
            OptimizationPriority::High,
            "Test Optimization",
            "Test description",
        )
        .with_component("test")
        .with_metrics("100ms", "50ms")
        .with_estimates(3, 5);

        assert_eq!(rec.priority, OptimizationPriority::High);
        assert_eq!(rec.title, "Test Optimization");
        assert_eq!(rec.effort_estimate, 3);
        assert!((rec.roi_score() - 5.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(OptimizationPriority::Critical > OptimizationPriority::High);
        assert!(OptimizationPriority::High > OptimizationPriority::Medium);
        assert!(OptimizationPriority::Medium > OptimizationPriority::Low);
    }

    #[test]
    fn test_report_metrics() {
        let metrics = ReportMetrics {
            total_constraints: 100000,
            constraint_target: 500000,
            total_time: Duration::from_millis(300),
            time_target: Duration::from_millis(500),
            memory_usage: 50_000_000,
            memory_target: 100_000_000,
            k_parameter: 14,
            proof_size: 1024,
        };

        assert!(metrics.total_constraints < metrics.constraint_target);
        assert!(metrics.total_time < metrics.time_target);
    }

    #[test]
    fn test_fix_risk() {
        assert!(matches!(FixRisk::None, FixRisk::None));
        assert!(matches!(FixRisk::High, FixRisk::High));
    }

    #[test]
    fn test_bottleneck_category() {
        assert!(matches!(BottleneckCategory::Constraints, BottleneckCategory::Constraints));
        assert!(matches!(BottleneckCategory::Timing, BottleneckCategory::Timing));
    }

    #[test]
    fn test_potential_savings() {
        let savings = PotentialSavings {
            constraint_reduction: 10000,
            time_reduction: Duration::from_millis(100),
            memory_reduction: 1_000_000,
            proof_size_reduction: 512,
        };

        assert_eq!(savings.constraint_reduction, 10000);
        assert_eq!(savings.time_reduction, Duration::from_millis(100));
    }
}
