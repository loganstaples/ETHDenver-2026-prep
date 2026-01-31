//! Real-time GPU vs CPU Profiler for HELIX.
//!
//! Provides live performance monitoring during proof generation:
//! - Operation timing with automatic GPU/CPU classification
//! - Memory usage tracking
//! - Throughput monitoring
//! - Live speedup calculation
//! - Profile export for analysis

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant};

/// Operation types for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Multi-scalar multiplication.
    Msm,
    /// Number-theoretic transform.
    Ntt,
    /// Field multiplication.
    FieldMul,
    /// Field addition.
    FieldAdd,
    /// Field inversion.
    FieldInv,
    /// Point doubling.
    PointDouble,
    /// Point addition.
    PointAdd,
    /// Memory transfer to GPU.
    MemoryToGpu,
    /// Memory transfer from GPU.
    MemoryFromGpu,
    /// Polynomial evaluation.
    PolyEval,
    /// Commitment generation.
    Commitment,
    /// Full proof generation.
    ProofGen,
    /// Custom operation.
    Custom,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpType::Msm => write!(f, "MSM"),
            OpType::Ntt => write!(f, "NTT"),
            OpType::FieldMul => write!(f, "FieldMul"),
            OpType::FieldAdd => write!(f, "FieldAdd"),
            OpType::FieldInv => write!(f, "FieldInv"),
            OpType::PointDouble => write!(f, "PointDouble"),
            OpType::PointAdd => write!(f, "PointAdd"),
            OpType::MemoryToGpu => write!(f, "MemToGPU"),
            OpType::MemoryFromGpu => write!(f, "MemFromGPU"),
            OpType::PolyEval => write!(f, "PolyEval"),
            OpType::Commitment => write!(f, "Commit"),
            OpType::ProofGen => write!(f, "ProofGen"),
            OpType::Custom => write!(f, "Custom"),
        }
    }
}

/// Backend type for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "CPU"),
            Backend::Metal => write!(f, "Metal"),
            Backend::Cuda => write!(f, "CUDA"),
        }
    }
}

/// Single profiled operation.
#[derive(Debug, Clone)]
pub struct ProfiledOp {
    /// Operation type.
    pub op_type: OpType,
    /// Backend used.
    pub backend: Backend,
    /// Duration.
    pub duration: Duration,
    /// Input size (elements).
    pub input_size: usize,
    /// Memory used (bytes).
    pub memory_bytes: u64,
    /// Timestamp when operation started.
    pub timestamp: Instant,
    /// Operation-specific metadata.
    pub metadata: HashMap<String, String>,
}

impl ProfiledOp {
    /// Returns throughput in elements per second.
    pub fn throughput(&self) -> f64 {
        if self.duration.as_nanos() == 0 {
            return 0.0;
        }
        self.input_size as f64 / self.duration.as_secs_f64()
    }

    /// Returns memory bandwidth in bytes per second.
    pub fn bandwidth(&self) -> f64 {
        if self.duration.as_nanos() == 0 {
            return 0.0;
        }
        self.memory_bytes as f64 / self.duration.as_secs_f64()
    }
}

/// Aggregated statistics for an operation type.
#[derive(Debug, Clone, Default)]
pub struct OpStats {
    /// Total operations.
    pub count: u64,
    /// Total time.
    pub total_time: Duration,
    /// Minimum time.
    pub min_time: Duration,
    /// Maximum time.
    pub max_time: Duration,
    /// Total elements processed.
    pub total_elements: u64,
    /// Total memory used.
    pub total_memory: u64,
}

impl OpStats {
    /// Returns average time per operation.
    pub fn avg_time(&self) -> Duration {
        if self.count == 0 {
            return Duration::ZERO;
        }
        self.total_time / self.count as u32
    }

    /// Returns average throughput.
    pub fn avg_throughput(&self) -> f64 {
        if self.total_time.as_nanos() == 0 {
            return 0.0;
        }
        self.total_elements as f64 / self.total_time.as_secs_f64()
    }

    /// Merges another stats object.
    pub fn merge(&mut self, other: &OpStats) {
        self.count += other.count;
        self.total_time += other.total_time;
        self.min_time = std::cmp::min(self.min_time, other.min_time);
        self.max_time = std::cmp::max(self.max_time, other.max_time);
        self.total_elements += other.total_elements;
        self.total_memory += other.total_memory;
    }
}

/// Inner state for profiler.
struct ProfilerInner {
    /// All profiled operations.
    operations: Vec<ProfiledOp>,
    /// Stats by (OpType, Backend).
    stats: HashMap<(OpType, Backend), OpStats>,
    /// Session start time.
    start_time: Instant,
    /// Whether profiling is enabled.
    enabled: bool,
    /// Maximum operations to store.
    max_operations: usize,
}

impl ProfilerInner {
    fn new(max_operations: usize) -> Self {
        Self {
            operations: Vec::new(),
            stats: HashMap::new(),
            start_time: Instant::now(),
            enabled: true,
            max_operations,
        }
    }

    fn record(&mut self, op: ProfiledOp) {
        if !self.enabled {
            return;
        }

        // Update stats
        let key = (op.op_type, op.backend);
        let stats = self.stats.entry(key).or_insert_with(|| OpStats {
            min_time: Duration::MAX,
            ..Default::default()
        });

        stats.count += 1;
        stats.total_time += op.duration;
        stats.min_time = std::cmp::min(stats.min_time, op.duration);
        stats.max_time = std::cmp::max(stats.max_time, op.duration);
        stats.total_elements += op.input_size as u64;
        stats.total_memory += op.memory_bytes;

        // Store operation if within limit
        if self.operations.len() < self.max_operations {
            self.operations.push(op);
        }
    }
}

/// Thread-safe GPU profiler.
pub struct GpuProfiler {
    inner: Arc<RwLock<ProfilerInner>>,
    /// Total operations recorded.
    total_ops: AtomicU64,
    /// Total GPU time in nanoseconds.
    gpu_time_ns: AtomicU64,
    /// Total CPU time in nanoseconds.
    cpu_time_ns: AtomicU64,
    /// Enabled flag for fast checking.
    enabled: AtomicBool,
}

impl GpuProfiler {
    /// Creates a new profiler.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ProfilerInner::new(10000))),
            total_ops: AtomicU64::new(0),
            gpu_time_ns: AtomicU64::new(0),
            cpu_time_ns: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Creates with custom capacity.
    pub fn with_capacity(max_operations: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ProfilerInner::new(max_operations))),
            total_ops: AtomicU64::new(0),
            gpu_time_ns: AtomicU64::new(0),
            cpu_time_ns: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Enables profiling.
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::SeqCst);
        if let Ok(mut inner) = self.inner.write() {
            inner.enabled = true;
        }
    }

    /// Disables profiling.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
        if let Ok(mut inner) = self.inner.write() {
            inner.enabled = false;
        }
    }

    /// Returns whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Records a profiled operation.
    pub fn record(&self, op: ProfiledOp) {
        if !self.is_enabled() {
            return;
        }

        // Update atomic counters
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        let ns = op.duration.as_nanos() as u64;
        match op.backend {
            Backend::Cpu => {
                self.cpu_time_ns.fetch_add(ns, Ordering::Relaxed);
            }
            Backend::Metal | Backend::Cuda => {
                self.gpu_time_ns.fetch_add(ns, Ordering::Relaxed);
            }
        }

        // Record full operation
        if let Ok(mut inner) = self.inner.write() {
            inner.record(op);
        }
    }

    /// Starts a timed operation and returns a guard.
    pub fn start_op(&self, op_type: OpType, backend: Backend, input_size: usize) -> OpGuard {
        OpGuard {
            profiler: self.clone(),
            op_type,
            backend,
            input_size,
            memory_bytes: 0,
            start: Instant::now(),
            metadata: HashMap::new(),
            finished: false,
        }
    }

    /// Returns statistics for an operation type and backend.
    pub fn stats_for(&self, op_type: OpType, backend: Backend) -> Option<OpStats> {
        if let Ok(inner) = self.inner.read() {
            inner.stats.get(&(op_type, backend)).cloned()
        } else {
            None
        }
    }

    /// Returns all statistics.
    pub fn all_stats(&self) -> HashMap<(OpType, Backend), OpStats> {
        if let Ok(inner) = self.inner.read() {
            inner.stats.clone()
        } else {
            HashMap::new()
        }
    }

    /// Returns total operations recorded.
    pub fn total_operations(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }

    /// Returns total GPU time.
    pub fn total_gpu_time(&self) -> Duration {
        Duration::from_nanos(self.gpu_time_ns.load(Ordering::Relaxed))
    }

    /// Returns total CPU time.
    pub fn total_cpu_time(&self) -> Duration {
        Duration::from_nanos(self.cpu_time_ns.load(Ordering::Relaxed))
    }

    /// Returns current GPU vs CPU speedup.
    pub fn current_speedup(&self) -> Option<f64> {
        let gpu_ns = self.gpu_time_ns.load(Ordering::Relaxed);
        let cpu_ns = self.cpu_time_ns.load(Ordering::Relaxed);

        if gpu_ns == 0 || cpu_ns == 0 {
            return None;
        }

        Some(cpu_ns as f64 / gpu_ns as f64)
    }

    /// Calculates speedup for a specific operation type.
    pub fn speedup_for(&self, op_type: OpType) -> Option<f64> {
        let cpu_stats = self.stats_for(op_type, Backend::Cpu)?;

        // Try Metal first, then CUDA
        let gpu_stats = self.stats_for(op_type, Backend::Metal)
            .or_else(|| self.stats_for(op_type, Backend::Cuda))?;

        if gpu_stats.total_time.as_nanos() == 0 {
            return None;
        }

        Some(cpu_stats.total_time.as_secs_f64() / gpu_stats.total_time.as_secs_f64())
    }

    /// Returns a formatted report.
    pub fn report(&self) -> ProfileReport {
        let stats = self.all_stats();
        let session_duration = if let Ok(inner) = self.inner.read() {
            inner.start_time.elapsed()
        } else {
            Duration::ZERO
        };

        ProfileReport {
            total_operations: self.total_operations(),
            total_gpu_time: self.total_gpu_time(),
            total_cpu_time: self.total_cpu_time(),
            session_duration,
            stats,
            speedups: self.calculate_all_speedups(),
        }
    }

    fn calculate_all_speedups(&self) -> HashMap<OpType, f64> {
        let mut speedups = HashMap::new();

        for op_type in [
            OpType::Msm,
            OpType::Ntt,
            OpType::FieldMul,
            OpType::ProofGen,
        ] {
            if let Some(speedup) = self.speedup_for(op_type) {
                speedups.insert(op_type, speedup);
            }
        }

        speedups
    }

    /// Clears all recorded data.
    pub fn clear(&self) {
        self.total_ops.store(0, Ordering::SeqCst);
        self.gpu_time_ns.store(0, Ordering::SeqCst);
        self.cpu_time_ns.store(0, Ordering::SeqCst);

        if let Ok(mut inner) = self.inner.write() {
            inner.operations.clear();
            inner.stats.clear();
            inner.start_time = Instant::now();
        }
    }
}

impl Clone for GpuProfiler {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            total_ops: AtomicU64::new(self.total_ops.load(Ordering::Relaxed)),
            gpu_time_ns: AtomicU64::new(self.gpu_time_ns.load(Ordering::Relaxed)),
            cpu_time_ns: AtomicU64::new(self.cpu_time_ns.load(Ordering::Relaxed)),
            enabled: AtomicBool::new(self.enabled.load(Ordering::Relaxed)),
        }
    }
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Guard for automatic operation timing.
///
/// When dropped, automatically records the operation timing.
/// Call `finish()` explicitly if you need to ensure recording happens.
pub struct OpGuard {
    profiler: GpuProfiler,
    op_type: OpType,
    backend: Backend,
    input_size: usize,
    memory_bytes: u64,
    start: Instant,
    metadata: HashMap<String, String>,
    finished: bool,
}

impl OpGuard {
    /// Sets memory usage for this operation.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Finishes timing and records the operation.
    /// This consumes the guard and records immediately.
    pub fn finish(mut self) {
        self.record_internal();
        self.finished = true;
    }

    fn record_internal(&mut self) {
        let duration = self.start.elapsed();

        self.profiler.record(ProfiledOp {
            op_type: self.op_type,
            backend: self.backend,
            duration,
            input_size: self.input_size,
            memory_bytes: self.memory_bytes,
            timestamp: self.start,
            metadata: std::mem::take(&mut self.metadata),
        });
    }
}

impl Drop for OpGuard {
    fn drop(&mut self) {
        // Only record if not already finished
        if !self.finished {
            self.record_internal();
        }
    }
}

/// Profile report with all statistics.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Total operations.
    pub total_operations: u64,
    /// Total GPU time.
    pub total_gpu_time: Duration,
    /// Total CPU time.
    pub total_cpu_time: Duration,
    /// Session duration.
    pub session_duration: Duration,
    /// Statistics by operation and backend.
    pub stats: HashMap<(OpType, Backend), OpStats>,
    /// Calculated speedups.
    pub speedups: HashMap<OpType, f64>,
}

impl ProfileReport {
    /// Prints the report.
    pub fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════════╗");
        println!("║                      GPU PROFILER REPORT                             ║");
        println!("╠══════════════════════════════════════════════════════════════════════╣");
        println!("║ Session duration: {:>10.2?}                                       ║", self.session_duration);
        println!("║ Total operations: {:>10}                                         ║", self.total_operations);
        println!("║ GPU time:         {:>10.2?}                                       ║", self.total_gpu_time);
        println!("║ CPU time:         {:>10.2?}                                       ║", self.total_cpu_time);
        println!("╠══════════════════════════════════════════════════════════════════════╣");
        println!("║ {:^12} │ {:^8} │ {:^8} │ {:^10} │ {:^10} │ {:^10} ║",
            "Operation", "Backend", "Count", "Avg (ms)", "Total (ms)", "Throughput");
        println!("╠══════════════════════════════════════════════════════════════════════╣");

        let mut entries: Vec<_> = self.stats.iter().collect();
        entries.sort_by_key(|((op, _), _)| format!("{:?}", op));

        for ((op_type, backend), stats) in entries {
            let avg_ms = stats.avg_time().as_secs_f64() * 1000.0;
            let total_ms = stats.total_time.as_secs_f64() * 1000.0;
            let throughput = if stats.total_time.as_nanos() > 0 {
                format!("{:.0}/s", stats.avg_throughput())
            } else {
                "-".to_string()
            };

            println!("║ {:^12} │ {:^8} │ {:>8} │ {:>10.3} │ {:>10.1} │ {:>10} ║",
                op_type, backend, stats.count, avg_ms, total_ms, throughput);
        }

        if !self.speedups.is_empty() {
            println!("╠══════════════════════════════════════════════════════════════════════╣");
            println!("║ SPEEDUPS (GPU vs CPU)                                                ║");

            for (op_type, speedup) in &self.speedups {
                let status = if *speedup >= 10.0 { "✓" } else { "" };
                println!("║   {}: {:.1}x {}                                                     ║",
                    op_type, speedup, status);
            }
        }

        println!("╚══════════════════════════════════════════════════════════════════════╝");
    }

    /// Returns as JSON.
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");
        json.push_str(&format!("  \"session_duration_ms\": {:.3},\n",
            self.session_duration.as_secs_f64() * 1000.0));
        json.push_str(&format!("  \"total_operations\": {},\n", self.total_operations));
        json.push_str(&format!("  \"gpu_time_ms\": {:.3},\n",
            self.total_gpu_time.as_secs_f64() * 1000.0));
        json.push_str(&format!("  \"cpu_time_ms\": {:.3},\n",
            self.total_cpu_time.as_secs_f64() * 1000.0));

        json.push_str("  \"speedups\": {\n");
        for (i, (op, speedup)) in self.speedups.iter().enumerate() {
            json.push_str(&format!("    \"{}\": {:.2}", op, speedup));
            if i < self.speedups.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  },\n");

        json.push_str("  \"operations\": [\n");
        for (i, ((op, backend), stats)) in self.stats.iter().enumerate() {
            json.push_str(&format!("    {{\"op\": \"{}\", \"backend\": \"{}\", \"count\": {}, \"avg_ms\": {:.3}, \"total_ms\": {:.3}}}",
                op, backend, stats.count,
                stats.avg_time().as_secs_f64() * 1000.0,
                stats.total_time.as_secs_f64() * 1000.0));
            if i < self.stats.len() - 1 {
                json.push(',');
            }
            json.push('\n');
        }
        json.push_str("  ]\n");
        json.push_str("}\n");
        json
    }
}

/// Global profiler instance.
static GLOBAL_PROFILER: std::sync::OnceLock<GpuProfiler> = std::sync::OnceLock::new();

/// Gets the global profiler.
pub fn global_profiler() -> &'static GpuProfiler {
    GLOBAL_PROFILER.get_or_init(GpuProfiler::new)
}

/// Convenience macro for profiling an operation.
#[macro_export]
macro_rules! profile_op {
    ($op_type:expr, $backend:expr, $size:expr, $code:block) => {{
        let _guard = $crate::benchmarks::profiler::global_profiler()
            .start_op($op_type, $backend, $size);
        let result = $code;
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let profiler = GpuProfiler::new();

        profiler.record(ProfiledOp {
            op_type: OpType::Msm,
            backend: Backend::Metal,
            duration: Duration::from_millis(10),
            input_size: 1000,
            memory_bytes: 32000,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        });

        assert_eq!(profiler.total_operations(), 1);

        let stats = profiler.stats_for(OpType::Msm, Backend::Metal).unwrap();
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_op_guard() {
        let profiler = GpuProfiler::new();

        {
            let _guard = profiler.start_op(OpType::Ntt, Backend::Cpu, 4096);
            std::thread::sleep(Duration::from_millis(5));
        }

        assert_eq!(profiler.total_operations(), 1);
        let stats = profiler.stats_for(OpType::Ntt, Backend::Cpu).unwrap();
        assert!(stats.total_time >= Duration::from_millis(5));
    }

    #[test]
    fn test_speedup_calculation() {
        let profiler = GpuProfiler::new();

        // Record CPU operation
        profiler.record(ProfiledOp {
            op_type: OpType::Msm,
            backend: Backend::Cpu,
            duration: Duration::from_millis(100),
            input_size: 1000,
            memory_bytes: 0,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        });

        // Record GPU operation
        profiler.record(ProfiledOp {
            op_type: OpType::Msm,
            backend: Backend::Metal,
            duration: Duration::from_millis(10),
            input_size: 1000,
            memory_bytes: 0,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        });

        let speedup = profiler.speedup_for(OpType::Msm).unwrap();
        assert!((speedup - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_report() {
        let profiler = GpuProfiler::new();

        for _ in 0..5 {
            profiler.record(ProfiledOp {
                op_type: OpType::FieldMul,
                backend: Backend::Metal,
                duration: Duration::from_micros(100),
                input_size: 256,
                memory_bytes: 1024,
                timestamp: Instant::now(),
                metadata: HashMap::new(),
            });
        }

        let report = profiler.report();
        assert_eq!(report.total_operations, 5);
    }
}
