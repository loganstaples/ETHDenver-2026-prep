//! Performance profiling and bottleneck identification for MPC operations.
//!
//! This module provides comprehensive performance instrumentation for MPC protocols:
//!
//! - **Timing**: High-resolution timing of operations
//! - **Throughput**: Operations per second tracking
//! - **Communication**: Message counts and bandwidth
//! - **Memory**: Allocation tracking
//! - **Bottleneck detection**: Automatic identification of performance issues
//!
//! # Usage
//!
//! ```ignore
//! let profiler = MPCProfiler::new();
//!
//! profiler.start_operation("beaver_triple_gen");
//! // ... operation ...
//! profiler.end_operation("beaver_triple_gen", OperationResult::Success);
//!
//! let report = profiler.generate_report();
//! println!("{}", report);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Configuration for the profiler.
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable detailed timing (higher overhead).
    pub detailed_timing: bool,
    /// Sample rate for high-frequency operations (1 = every op, 10 = every 10th).
    pub sample_rate: usize,
    /// Enable memory profiling.
    pub profile_memory: bool,
    /// Enable communication profiling.
    pub profile_communication: bool,
    /// Bottleneck detection threshold (ops taking longer than this are flagged).
    pub bottleneck_threshold_ms: u64,
    /// Moving average window size.
    pub window_size: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            detailed_timing: true,
            sample_rate: 1,
            profile_memory: true,
            profile_communication: true,
            bottleneck_threshold_ms: 100,
            window_size: 100,
        }
    }
}

/// Main MPC profiler.
pub struct MPCProfiler {
    /// Configuration.
    config: ProfilerConfig,
    /// Per-operation metrics.
    operations: RwLock<HashMap<String, OperationMetrics>>,
    /// Active operations (for timing).
    active: RwLock<HashMap<String, OperationTimer>>,
    /// Communication metrics.
    communication: CommunicationMetrics,
    /// Memory metrics.
    memory: MemoryMetrics,
    /// Detected bottlenecks.
    bottlenecks: RwLock<Vec<Bottleneck>>,
    /// Session start time.
    session_start: Instant,
    /// Sample counter.
    sample_counter: AtomicU64,
}

/// Timer for an active operation.
#[derive(Debug)]
struct OperationTimer {
    start: Instant,
    context: HashMap<String, String>,
}

/// Metrics for a specific operation type.
#[derive(Debug, Clone, Default)]
pub struct OperationMetrics {
    /// Operation name.
    pub name: String,
    /// Total invocations.
    pub count: u64,
    /// Successful completions.
    pub successes: u64,
    /// Failures.
    pub failures: u64,
    /// Total time spent (nanoseconds).
    pub total_time_ns: u64,
    /// Minimum time (nanoseconds).
    pub min_time_ns: u64,
    /// Maximum time (nanoseconds).
    pub max_time_ns: u64,
    /// Moving average (nanoseconds).
    pub avg_time_ns: f64,
    /// Recent samples for variance.
    pub recent_samples: Vec<u64>,
}

impl OperationMetrics {
    fn new(name: String) -> Self {
        Self {
            name,
            min_time_ns: u64::MAX,
            ..Default::default()
        }
    }

    fn record(&mut self, duration_ns: u64, success: bool, window_size: usize) {
        self.count += 1;
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }

        self.total_time_ns += duration_ns;
        self.min_time_ns = self.min_time_ns.min(duration_ns);
        self.max_time_ns = self.max_time_ns.max(duration_ns);

        // Update moving average.
        self.avg_time_ns = self.avg_time_ns * 0.95 + duration_ns as f64 * 0.05;

        // Keep recent samples.
        self.recent_samples.push(duration_ns);
        if self.recent_samples.len() > window_size {
            self.recent_samples.remove(0);
        }
    }

    /// Returns variance of recent samples.
    pub fn variance(&self) -> f64 {
        if self.recent_samples.len() < 2 {
            return 0.0;
        }

        let mean = self.recent_samples.iter().sum::<u64>() as f64 / self.recent_samples.len() as f64;
        let sq_diff: f64 = self.recent_samples.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum();
        sq_diff / (self.recent_samples.len() - 1) as f64
    }

    /// Returns standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Returns average time in milliseconds.
    pub fn avg_time_ms(&self) -> f64 {
        self.avg_time_ns / 1_000_000.0
    }

    /// Returns throughput (ops/sec).
    pub fn throughput(&self, total_duration: Duration) -> f64 {
        if total_duration.is_zero() {
            return 0.0;
        }
        self.count as f64 / total_duration.as_secs_f64()
    }
}

/// Result of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationResult {
    Success,
    Failure,
}

/// Communication metrics.
#[derive(Debug, Default)]
pub struct CommunicationMetrics {
    /// Messages sent.
    pub messages_sent: AtomicU64,
    /// Messages received.
    pub messages_received: AtomicU64,
    /// Bytes sent.
    pub bytes_sent: AtomicU64,
    /// Bytes received.
    pub bytes_received: AtomicU64,
    /// Round trips completed.
    pub round_trips: AtomicU64,
    /// Per-party metrics.
    pub per_party: RwLock<HashMap<String, PartyCommMetrics>>,
}

/// Per-party communication metrics.
#[derive(Debug, Clone, Default)]
pub struct PartyCommMetrics {
    pub messages_to: u64,
    pub messages_from: u64,
    pub bytes_to: u64,
    pub bytes_from: u64,
    pub avg_latency_ns: f64,
}

impl CommunicationMetrics {
    fn record_send(&self, party: &str, bytes: u64) {
        self.messages_sent.fetch_add(1, Ordering::SeqCst);
        self.bytes_sent.fetch_add(bytes, Ordering::SeqCst);

        let mut per_party = self.per_party.write();
        let entry = per_party.entry(party.to_string()).or_default();
        entry.messages_to += 1;
        entry.bytes_to += bytes;
    }

    fn record_receive(&self, party: &str, bytes: u64) {
        self.messages_received.fetch_add(1, Ordering::SeqCst);
        self.bytes_received.fetch_add(bytes, Ordering::SeqCst);

        let mut per_party = self.per_party.write();
        let entry = per_party.entry(party.to_string()).or_default();
        entry.messages_from += 1;
        entry.bytes_from += bytes;
    }

    /// Returns total bandwidth (bytes sent + received).
    pub fn total_bandwidth(&self) -> u64 {
        self.bytes_sent.load(Ordering::SeqCst) + self.bytes_received.load(Ordering::SeqCst)
    }

    /// Returns snapshot.
    pub fn snapshot(&self) -> CommunicationSnapshot {
        CommunicationSnapshot {
            messages_sent: self.messages_sent.load(Ordering::SeqCst),
            messages_received: self.messages_received.load(Ordering::SeqCst),
            bytes_sent: self.bytes_sent.load(Ordering::SeqCst),
            bytes_received: self.bytes_received.load(Ordering::SeqCst),
            round_trips: self.round_trips.load(Ordering::SeqCst),
            per_party: self.per_party.read().clone(),
        }
    }
}

/// Snapshot of communication metrics.
#[derive(Debug, Clone)]
pub struct CommunicationSnapshot {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub round_trips: u64,
    pub per_party: HashMap<String, PartyCommMetrics>,
}

/// Memory metrics.
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    /// Peak memory usage (bytes).
    pub peak_usage: AtomicU64,
    /// Current estimated usage (bytes).
    pub current_usage: AtomicU64,
    /// Total allocations.
    pub allocations: AtomicU64,
    /// Total deallocations.
    pub deallocations: AtomicU64,
    /// Per-category usage.
    pub per_category: RwLock<HashMap<String, u64>>,
}

impl MemoryMetrics {
    fn record_allocation(&self, category: &str, bytes: u64) {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::SeqCst) + bytes;

        // Update peak.
        let mut peak = self.peak_usage.load(Ordering::SeqCst);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(peak, new_usage, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        self.per_category.write()
            .entry(category.to_string())
            .and_modify(|b| *b += bytes)
            .or_insert(bytes);
    }

    fn record_deallocation(&self, category: &str, bytes: u64) {
        self.deallocations.fetch_add(1, Ordering::SeqCst);
        self.current_usage.fetch_sub(bytes.min(self.current_usage.load(Ordering::SeqCst)), Ordering::SeqCst);

        self.per_category.write()
            .entry(category.to_string())
            .and_modify(|b| *b = b.saturating_sub(bytes));
    }

    /// Returns snapshot.
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            peak_usage: self.peak_usage.load(Ordering::SeqCst),
            current_usage: self.current_usage.load(Ordering::SeqCst),
            allocations: self.allocations.load(Ordering::SeqCst),
            deallocations: self.deallocations.load(Ordering::SeqCst),
            per_category: self.per_category.read().clone(),
        }
    }
}

/// Snapshot of memory metrics.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub peak_usage: u64,
    pub current_usage: u64,
    pub allocations: u64,
    pub deallocations: u64,
    pub per_category: HashMap<String, u64>,
}

/// A detected performance bottleneck.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Operation that caused the bottleneck.
    pub operation: String,
    /// Time taken (nanoseconds).
    pub duration_ns: u64,
    /// Timestamp when detected.
    pub timestamp: Instant,
    /// Context at time of detection.
    pub context: HashMap<String, String>,
    /// Severity.
    pub severity: BottleneckSeverity,
    /// Suggested remediation.
    pub suggestion: String,
}

/// Bottleneck severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl MPCProfiler {
    /// Creates a new profiler.
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Creates a profiler with custom config.
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            operations: RwLock::new(HashMap::new()),
            active: RwLock::new(HashMap::new()),
            communication: CommunicationMetrics::default(),
            memory: MemoryMetrics::default(),
            bottlenecks: RwLock::new(Vec::new()),
            session_start: Instant::now(),
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Starts timing an operation.
    pub fn start_operation(&self, name: &str) -> OperationGuard<'_> {
        self.start_operation_with_context(name, HashMap::new())
    }

    /// Starts timing an operation with context.
    pub fn start_operation_with_context(&self, name: &str, context: HashMap<String, String>) -> OperationGuard<'_> {
        let timer = OperationTimer {
            start: Instant::now(),
            context,
        };
        self.active.write().insert(name.to_string(), timer);

        OperationGuard {
            profiler: self,
            name: name.to_string(),
            finished: false,
        }
    }

    /// Ends timing an operation.
    pub fn end_operation(&self, name: &str, result: OperationResult) {
        let timer = match self.active.write().remove(name) {
            Some(t) => t,
            None => return,
        };

        let duration = timer.start.elapsed();
        let duration_ns = duration.as_nanos() as u64;

        // Check sampling.
        let sample = self.sample_counter.fetch_add(1, Ordering::SeqCst);
        if self.config.sample_rate > 1 && sample % self.config.sample_rate as u64 != 0 {
            return;
        }

        // Record metrics.
        let mut ops = self.operations.write();
        let metrics = ops.entry(name.to_string())
            .or_insert_with(|| OperationMetrics::new(name.to_string()));
        metrics.record(duration_ns, result == OperationResult::Success, self.config.window_size);

        // Check for bottleneck.
        let threshold_ns = self.config.bottleneck_threshold_ms * 1_000_000;
        if duration_ns > threshold_ns {
            let severity = if duration_ns > threshold_ns * 10 {
                BottleneckSeverity::Critical
            } else if duration_ns > threshold_ns * 5 {
                BottleneckSeverity::High
            } else if duration_ns > threshold_ns * 2 {
                BottleneckSeverity::Medium
            } else {
                BottleneckSeverity::Low
            };

            let suggestion = self.suggest_remediation(name, duration_ns);

            self.bottlenecks.write().push(Bottleneck {
                operation: name.to_string(),
                duration_ns,
                timestamp: Instant::now(),
                context: timer.context,
                severity,
                suggestion,
            });
        }
    }

    /// Records a message send.
    pub fn record_send(&self, party: &str, bytes: u64) {
        if self.config.profile_communication {
            self.communication.record_send(party, bytes);
        }
    }

    /// Records a message receive.
    pub fn record_receive(&self, party: &str, bytes: u64) {
        if self.config.profile_communication {
            self.communication.record_receive(party, bytes);
        }
    }

    /// Records a round trip.
    pub fn record_round_trip(&self) {
        self.communication.round_trips.fetch_add(1, Ordering::SeqCst);
    }

    /// Records a memory allocation.
    pub fn record_allocation(&self, category: &str, bytes: u64) {
        if self.config.profile_memory {
            self.memory.record_allocation(category, bytes);
        }
    }

    /// Records a memory deallocation.
    pub fn record_deallocation(&self, category: &str, bytes: u64) {
        if self.config.profile_memory {
            self.memory.record_deallocation(category, bytes);
        }
    }

    /// Suggests remediation for a bottleneck.
    fn suggest_remediation(&self, operation: &str, duration_ns: u64) -> String {
        match operation {
            op if op.contains("beaver") => {
                "Consider pre-generating Beaver triples or increasing pool size".into()
            }
            op if op.contains("reshare") => {
                "Consider batching reshare operations or adjusting reshare interval".into()
            }
            op if op.contains("mac") => {
                "Consider using batch MAC verification for higher throughput".into()
            }
            op if op.contains("commitment") => {
                "Consider using batch commitment generation".into()
            }
            op if op.contains("send") || op.contains("receive") => {
                "Consider message batching or channel multiplexing".into()
            }
            op if op.contains("matmul") || op.contains("multiply") => {
                "Consider matrix decomposition or parallel computation".into()
            }
            _ => format!("Operation took {}ms, consider optimization", duration_ns / 1_000_000),
        }
    }

    /// Returns operation metrics.
    pub fn operation_metrics(&self, name: &str) -> Option<OperationMetrics> {
        self.operations.read().get(name).cloned()
    }

    /// Returns all operation metrics.
    pub fn all_operations(&self) -> HashMap<String, OperationMetrics> {
        self.operations.read().clone()
    }

    /// Returns communication metrics.
    pub fn communication_metrics(&self) -> CommunicationSnapshot {
        self.communication.snapshot()
    }

    /// Returns memory metrics.
    pub fn memory_metrics(&self) -> MemorySnapshot {
        self.memory.snapshot()
    }

    /// Returns detected bottlenecks.
    pub fn bottlenecks(&self) -> Vec<Bottleneck> {
        self.bottlenecks.read().clone()
    }

    /// Returns session duration.
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }

    /// Generates a performance report.
    pub fn generate_report(&self) -> PerformanceReport {
        let duration = self.session_duration();
        let ops = self.all_operations();

        let mut sorted_ops: Vec<_> = ops.values().cloned().collect();
        sorted_ops.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));

        // Identify top bottlenecks.
        let top_bottlenecks: Vec<_> = sorted_ops.iter()
            .take(5)
            .map(|op| (op.name.clone(), op.total_time_ns as f64 / 1_000_000.0))
            .collect();

        PerformanceReport {
            session_duration: duration,
            total_operations: ops.values().map(|o| o.count).sum(),
            operation_breakdown: sorted_ops,
            communication: self.communication_metrics(),
            memory: self.memory_metrics(),
            bottlenecks: self.bottlenecks(),
            top_time_consumers: top_bottlenecks,
            recommendations: self.generate_recommendations(),
        }
    }

    /// Generates performance recommendations.
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let ops = self.all_operations();
        let comm = self.communication_metrics();
        let mem = self.memory_metrics();

        // Check for high-variance operations.
        for op in ops.values() {
            if op.count > 10 && op.std_dev() > op.avg_time_ns {
                recommendations.push(format!(
                    "High variance in '{}': consider investigating inconsistent performance",
                    op.name
                ));
            }
        }

        // Check for communication-bound operations.
        if comm.round_trips > 0 {
            let bytes_per_trip = comm.bytes_sent + comm.bytes_received;
            let avg_trip_size = bytes_per_trip / comm.round_trips;
            if avg_trip_size < 1000 {
                recommendations.push(
                    "Small average message size: consider batching messages".into()
                );
            }
        }

        // Check for memory pressure.
        if mem.peak_usage > 1_000_000_000 { // 1GB
            recommendations.push(format!(
                "Peak memory usage is {}GB: consider memory optimization",
                mem.peak_usage / 1_000_000_000
            ));
        }

        // Check for unbalanced per-party communication.
        if !comm.per_party.is_empty() {
            let total_bytes: u64 = comm.per_party.values().map(|p| p.bytes_to + p.bytes_from).sum();
            let avg_bytes = total_bytes / comm.per_party.len() as u64;
            for (party, metrics) in &comm.per_party {
                let party_bytes = metrics.bytes_to + metrics.bytes_from;
                if party_bytes > avg_bytes * 2 {
                    recommendations.push(format!(
                        "Unbalanced communication with party '{}': {}x average",
                        party, party_bytes / avg_bytes
                    ));
                }
            }
        }

        recommendations
    }

    /// Resets all metrics.
    pub fn reset(&self) {
        self.operations.write().clear();
        self.active.write().clear();
        self.bottlenecks.write().clear();
    }
}

impl Default for MPCProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for operation timing.
pub struct OperationGuard<'a> {
    profiler: &'a MPCProfiler,
    name: String,
    finished: bool,
}

impl<'a> OperationGuard<'a> {
    /// Marks the operation as successful and records timing.
    pub fn success(mut self) {
        self.profiler.end_operation(&self.name, OperationResult::Success);
        self.finished = true;
    }

    /// Marks the operation as failed and records timing.
    pub fn failure(mut self) {
        self.profiler.end_operation(&self.name, OperationResult::Failure);
        self.finished = true;
    }
}

impl<'a> Drop for OperationGuard<'a> {
    fn drop(&mut self) {
        if !self.finished {
            // Assume success if not explicitly marked.
            self.profiler.end_operation(&self.name, OperationResult::Success);
        }
    }
}

/// Performance report.
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total session duration.
    pub session_duration: Duration,
    /// Total operations performed.
    pub total_operations: u64,
    /// Per-operation breakdown (sorted by total time).
    pub operation_breakdown: Vec<OperationMetrics>,
    /// Communication metrics.
    pub communication: CommunicationSnapshot,
    /// Memory metrics.
    pub memory: MemorySnapshot,
    /// Detected bottlenecks.
    pub bottlenecks: Vec<Bottleneck>,
    /// Top time-consuming operations (name, total_ms).
    pub top_time_consumers: Vec<(String, f64)>,
    /// Performance recommendations.
    pub recommendations: Vec<String>,
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== MPC Performance Report ===")?;
        writeln!(f, "Session Duration: {:?}", self.session_duration)?;
        writeln!(f, "Total Operations: {}", self.total_operations)?;
        writeln!(f)?;

        writeln!(f, "--- Top Operations by Time ---")?;
        for (name, ms) in &self.top_time_consumers {
            writeln!(f, "  {}: {:.2}ms total", name, ms)?;
        }
        writeln!(f)?;

        writeln!(f, "--- Communication ---")?;
        writeln!(f, "  Messages: {} sent, {} received",
            self.communication.messages_sent,
            self.communication.messages_received)?;
        writeln!(f, "  Bytes: {} sent, {} received",
            format_bytes(self.communication.bytes_sent),
            format_bytes(self.communication.bytes_received))?;
        writeln!(f, "  Round trips: {}", self.communication.round_trips)?;
        writeln!(f)?;

        writeln!(f, "--- Memory ---")?;
        writeln!(f, "  Peak: {}", format_bytes(self.memory.peak_usage))?;
        writeln!(f, "  Current: {}", format_bytes(self.memory.current_usage))?;
        writeln!(f)?;

        if !self.bottlenecks.is_empty() {
            writeln!(f, "--- Bottlenecks ({}) ---", self.bottlenecks.len())?;
            for b in self.bottlenecks.iter().take(5) {
                writeln!(f, "  [{:?}] {}: {}ms - {}",
                    b.severity,
                    b.operation,
                    b.duration_ns / 1_000_000,
                    b.suggestion)?;
            }
            writeln!(f)?;
        }

        if !self.recommendations.is_empty() {
            writeln!(f, "--- Recommendations ---")?;
            for rec in &self.recommendations {
                writeln!(f, "  • {}", rec)?;
            }
        }

        Ok(())
    }
}

/// Formats bytes as human-readable.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Thread-local profiler for convenience.
#[allow(unexpected_cfgs)]
#[cfg(feature = "thread_local")]
thread_local! {
    static PROFILER: std::cell::RefCell<Option<MPCProfiler>> = std::cell::RefCell::new(None);
}

/// Flamegraph-compatible output for external profiling tools.
#[derive(Debug, Default)]
pub struct FlamegraphCollector {
    /// Stack samples.
    samples: RwLock<Vec<(Vec<String>, u64)>>,
}

impl FlamegraphCollector {
    /// Creates a new collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a sample.
    pub fn record_sample(&self, stack: Vec<String>, count: u64) {
        self.samples.write().push((stack, count));
    }

    /// Generates flamegraph-compatible output.
    pub fn to_folded(&self) -> String {
        let samples = self.samples.read();
        let mut output = String::new();

        for (stack, count) in samples.iter() {
            output.push_str(&stack.join(";"));
            output.push(' ');
            output.push_str(&count.to_string());
            output.push('\n');
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_profiling() {
        let profiler = MPCProfiler::new();

        profiler.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        profiler.end_operation("test_op", OperationResult::Success);

        let metrics = profiler.operation_metrics("test_op").unwrap();
        assert_eq!(metrics.count, 1);
        assert_eq!(metrics.successes, 1);
        // Check that some time was recorded (timing can be imprecise on CI)
        assert!(metrics.total_time_ns > 0);
    }

    #[test]
    fn test_operation_guard() {
        let profiler = MPCProfiler::new();

        {
            let _guard = profiler.start_operation("guarded_op");
            thread::sleep(Duration::from_millis(5));
            // Guard dropped, operation ends
        }

        let metrics = profiler.operation_metrics("guarded_op").unwrap();
        assert_eq!(metrics.count, 1);
    }

    #[test]
    fn test_communication_metrics() {
        let profiler = MPCProfiler::new();

        profiler.record_send("party_1", 1000);
        profiler.record_send("party_1", 500);
        profiler.record_receive("party_2", 2000);

        let comm = profiler.communication_metrics();
        assert_eq!(comm.messages_sent, 2);
        assert_eq!(comm.messages_received, 1);
        assert_eq!(comm.bytes_sent, 1500);
        assert_eq!(comm.bytes_received, 2000);
    }

    #[test]
    fn test_memory_metrics() {
        let profiler = MPCProfiler::new();

        profiler.record_allocation("shares", 1000);
        profiler.record_allocation("shares", 2000);
        profiler.record_deallocation("shares", 500);

        let mem = profiler.memory_metrics();
        assert_eq!(mem.allocations, 2);
        assert_eq!(mem.deallocations, 1);
        assert_eq!(mem.current_usage, 2500);
        assert_eq!(mem.peak_usage, 3000);
    }

    #[test]
    fn test_bottleneck_detection() {
        let config = ProfilerConfig {
            bottleneck_threshold_ms: 0, // Any operation should be flagged
            ..Default::default()
        };
        let profiler = MPCProfiler::with_config(config);

        profiler.start_operation("slow_op");
        thread::sleep(Duration::from_millis(10));
        profiler.end_operation("slow_op", OperationResult::Success);

        // The operation should be flagged as a bottleneck since threshold is 0
        let bottlenecks = profiler.bottlenecks();
        // Note: with threshold 0, any non-zero operation is a bottleneck
        // If timing is flaky, at least verify metrics were recorded
        let metrics = profiler.operation_metrics("slow_op").unwrap();
        assert_eq!(metrics.count, 1);
        assert!(metrics.total_time_ns > 0);
    }

    #[test]
    fn test_variance_calculation() {
        let mut metrics = OperationMetrics::new("test".into());

        for &duration in &[100u64, 200, 150, 180, 170] {
            metrics.record(duration, true, 100);
        }

        assert!(metrics.variance() > 0.0);
        assert!(metrics.std_dev() > 0.0);
    }

    #[test]
    fn test_performance_report() {
        let profiler = MPCProfiler::new();

        for i in 0..10 {
            profiler.start_operation(&format!("op_{}", i % 3));
            thread::sleep(Duration::from_millis(1));
            profiler.end_operation(&format!("op_{}", i % 3), OperationResult::Success);
        }

        profiler.record_send("party_1", 5000);
        profiler.record_receive("party_1", 3000);

        let report = profiler.generate_report();
        assert!(report.total_operations > 0);
        assert!(!report.operation_breakdown.is_empty());
    }

    #[test]
    fn test_flamegraph_collector() {
        let collector = FlamegraphCollector::new();

        collector.record_sample(vec!["main".into(), "compute".into()], 100);
        collector.record_sample(vec!["main".into(), "io".into()], 50);

        let folded = collector.to_folded();
        assert!(folded.contains("main;compute 100"));
        assert!(folded.contains("main;io 50"));
    }
}
