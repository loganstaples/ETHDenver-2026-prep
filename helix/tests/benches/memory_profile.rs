//! Memory Usage Profiling
//!
//! This module provides memory profiling for HELIX proving operations.
//! Understanding memory usage is critical for:
//! - Determining maximum model sizes that can be proven
//! - Optimizing batch sizes for proving
//! - Planning hardware requirements for workers
//!
//! # Metrics Collected
//!
//! - Peak memory allocation
//! - Proof memory (resident size during proving)
//! - Memory per operation type
//! - Memory scaling with model size

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_prover::gkr::{GKRProver, GKRConfig, LayeredCircuit};
use serde::{Deserialize, Serialize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::{BenchmarkHarness, BenchmarkMetrics, ModelSize, CircuitSize, TimingHelper};
use super::gkr_prover::{create_benchmark_circuit, create_mlp_circuit};

/// Global allocator wrapper for tracking memory usage.
#[global_allocator]
static ALLOCATOR: MemoryTracker = MemoryTracker;

/// Custom allocator that tracks memory usage.
pub struct MemoryTracker;

/// Current allocated bytes (atomic for thread safety).
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Peak allocated bytes.
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for MemoryTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
            // Update peak if this is a new high
            let mut peak = PEAK.load(Ordering::SeqCst);
            while current > peak {
                match PEAK.compare_exchange(peak, current, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            if new_size > old_size {
                let diff = new_size - old_size;
                let current = ALLOCATED.fetch_add(diff, Ordering::SeqCst) + diff;
                let mut peak = PEAK.load(Ordering::SeqCst);
                while current > peak {
                    match PEAK.compare_exchange(peak, current, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }
            } else {
                ALLOCATED.fetch_sub(old_size - new_size, Ordering::SeqCst);
            }
        }
        new_ptr
    }
}

/// Memory snapshot.
#[derive(Debug, Clone, Copy)]
pub struct MemorySnapshot {
    pub allocated: usize,
    pub peak: usize,
}

impl MemorySnapshot {
    /// Takes a current memory snapshot.
    pub fn now() -> Self {
        Self {
            allocated: ALLOCATED.load(Ordering::SeqCst),
            peak: PEAK.load(Ordering::SeqCst),
        }
    }

    /// Resets the peak counter.
    pub fn reset_peak() {
        let current = ALLOCATED.load(Ordering::SeqCst);
        PEAK.store(current, Ordering::SeqCst);
    }

    /// Returns allocated in MB.
    pub fn allocated_mb(&self) -> f64 {
        self.allocated as f64 / (1024.0 * 1024.0)
    }

    /// Returns peak in MB.
    pub fn peak_mb(&self) -> f64 {
        self.peak as f64 / (1024.0 * 1024.0)
    }
}

/// Memory profile for a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub operation: String,
    pub before_bytes: usize,
    pub after_bytes: usize,
    pub peak_bytes: usize,
    pub delta_bytes: i64,
    pub duration_us: u64,
}

impl MemoryProfile {
    /// Formats memory in human-readable form.
    pub fn format_bytes(bytes: usize) -> String {
        if bytes < 1024 {
            format!("{} B", bytes)
        } else if bytes < 1024 * 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else if bytes < 1024 * 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }

    pub fn peak_formatted(&self) -> String {
        Self::format_bytes(self.peak_bytes)
    }

    pub fn delta_formatted(&self) -> String {
        if self.delta_bytes >= 0 {
            format!("+{}", Self::format_bytes(self.delta_bytes as usize))
        } else {
            format!("-{}", Self::format_bytes((-self.delta_bytes) as usize))
        }
    }
}

/// Memory profiler for running operations and collecting profiles.
pub struct MemoryProfiler {
    profiles: Vec<MemoryProfile>,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Profiles a function and records memory usage.
    pub fn profile<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Force garbage collection if possible
        // (Rust doesn't have GC, but we can encourage cleanup)

        let before = MemorySnapshot::now();
        MemorySnapshot::reset_peak();

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        let after = MemorySnapshot::now();

        self.profiles.push(MemoryProfile {
            operation: name.to_string(),
            before_bytes: before.allocated,
            after_bytes: after.allocated,
            peak_bytes: after.peak,
            delta_bytes: after.allocated as i64 - before.allocated as i64,
            duration_us: duration.as_micros() as u64,
        });

        result
    }

    /// Returns all collected profiles.
    pub fn profiles(&self) -> &[MemoryProfile] {
        &self.profiles
    }

    /// Generates a summary report.
    pub fn summary(&self) -> MemoryProfileSummary {
        if self.profiles.is_empty() {
            return MemoryProfileSummary::default();
        }

        let total_peak: usize = self.profiles.iter().map(|p| p.peak_bytes).max().unwrap_or(0);
        let avg_peak: f64 = self.profiles.iter().map(|p| p.peak_bytes as f64).sum::<f64>()
            / self.profiles.len() as f64;

        MemoryProfileSummary {
            num_operations: self.profiles.len(),
            total_peak_bytes: total_peak,
            avg_peak_bytes: avg_peak as usize,
            profiles: self.profiles.clone(),
        }
    }

    /// Prints profiles to terminal.
    pub fn print_summary(&self) {
        println!("\nMemory Profile Summary");
        println!("{}", "=".repeat(80));
        println!(
            "{:<30} {:>12} {:>12} {:>12} {:>10}",
            "Operation", "Before", "Peak", "After", "Duration"
        );
        println!("{}", "-".repeat(80));

        for profile in &self.profiles {
            println!(
                "{:<30} {:>12} {:>12} {:>12} {:>10}",
                profile.operation,
                MemoryProfile::format_bytes(profile.before_bytes),
                MemoryProfile::format_bytes(profile.peak_bytes),
                MemoryProfile::format_bytes(profile.after_bytes),
                format!("{:.1}ms", profile.duration_us as f64 / 1000.0),
            );
        }

        if let Some(max_peak) = self.profiles.iter().map(|p| p.peak_bytes).max() {
            println!("{}", "-".repeat(80));
            println!(
                "Maximum peak memory: {}",
                MemoryProfile::format_bytes(max_peak)
            );
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of memory profiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryProfileSummary {
    pub num_operations: usize,
    pub total_peak_bytes: usize,
    pub avg_peak_bytes: usize,
    pub profiles: Vec<MemoryProfile>,
}

/// Profiles GKR proving at different sizes.
pub fn profile_gkr_memory(profiler: &mut MemoryProfiler) {
    // Profile circuit creation
    for &size in &[CircuitSize::Small, CircuitSize::Medium] {
        let gates = size.gate_count();
        profiler.profile(&format!("gkr_circuit_{}gates", gates), || {
            create_benchmark_circuit(gates)
        });
    }

    // Profile proving at different sizes
    for &size in &[CircuitSize::XSmall, CircuitSize::Small] {
        let gates = size.gate_count();
        let circuit = create_benchmark_circuit(gates);
        let inputs = vec![Fr::from(3u64), Fr::from(5u64)];
        let config = GKRConfig::default();

        profiler.profile(&format!("gkr_prove_{}gates", gates), || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }
}

/// Profiles MLP proving at different model sizes.
pub fn profile_mlp_memory(profiler: &mut MemoryProfiler) {
    for &size in &[ModelSize::Tiny, ModelSize::Small, ModelSize::Medium] {
        let (d_in, d_hid, d_out) = size.dimensions();
        let inputs: Vec<Fr> = (0..d_in).map(|i| Fr::from(i as u64 + 1)).collect();
        let config = GKRConfig::default();

        // Profile circuit creation
        profiler.profile(&format!("mlp_circuit_{}", size.name()), || {
            create_mlp_circuit(d_in, d_hid, d_out)
        });

        // Profile proving
        let circuit = create_mlp_circuit(d_in, d_hid, d_out);
        profiler.profile(&format!("mlp_prove_{}", size.name()), || {
            let mut prover = GKRProver::new(config.clone());
            prover.prove(&circuit, &inputs).unwrap()
        });
    }
}

/// Estimates memory requirements for a given model size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub model_size: String,
    pub param_count: usize,
    pub estimated_circuit_memory: usize,
    pub estimated_proving_memory: usize,
    pub estimated_total_memory: usize,
}

impl MemoryEstimate {
    /// Estimates memory requirements for a model size.
    /// These are rough estimates based on observed behavior.
    pub fn estimate(size: ModelSize) -> Self {
        let params = size.param_count();

        // Circuit memory: ~100 bytes per gate, ~5 gates per parameter
        let gates = params * 5;
        let circuit_memory = gates * 100;

        // Proving memory: ~500 bytes per gate during proving
        let proving_memory = gates * 500;

        Self {
            model_size: size.to_string(),
            param_count: params,
            estimated_circuit_memory: circuit_memory,
            estimated_proving_memory: proving_memory,
            estimated_total_memory: circuit_memory + proving_memory,
        }
    }

    /// Estimates for all model sizes.
    pub fn estimate_all() -> Vec<Self> {
        ModelSize::all()
            .iter()
            .map(|&size| Self::estimate(size))
            .collect()
    }

    /// Prints a memory requirements table.
    pub fn print_table(estimates: &[Self]) {
        println!("\nMemory Requirements Estimate");
        println!("{}", "=".repeat(80));
        println!(
            "{:<20} {:>12} {:>15} {:>15} {:>15}",
            "Model", "Params", "Circuit", "Proving", "Total"
        );
        println!("{}", "-".repeat(80));

        for est in estimates {
            println!(
                "{:<20} {:>12} {:>15} {:>15} {:>15}",
                est.model_size,
                est.param_count,
                MemoryProfile::format_bytes(est.estimated_circuit_memory),
                MemoryProfile::format_bytes(est.estimated_proving_memory),
                MemoryProfile::format_bytes(est.estimated_total_memory),
            );
        }
    }
}

/// Runs memory profiling benchmarks.
pub fn run_memory_profiling(harness: &mut BenchmarkHarness) {
    let mut profiler = MemoryProfiler::new();

    // Profile GKR proving
    profile_gkr_memory(&mut profiler);

    // Profile MLP proving
    profile_mlp_memory(&mut profiler);

    // Print summary
    profiler.print_summary();

    // Record metrics in harness
    for profile in profiler.profiles() {
        let metrics = BenchmarkMetrics {
            mean_ns: profile.duration_us as f64 * 1000.0,
            stddev_ns: 0.0,
            min_ns: profile.duration_us as f64 * 1000.0,
            max_ns: profile.duration_us as f64 * 1000.0,
            iterations: 1,
            throughput_ops: None,
            memory_bytes: Some(profile.peak_bytes),
            tags: BTreeMap::new(),
        };
        harness.record_benchmark(&profile.operation, "memory", metrics);
    }
}

/// Gets current memory usage for CI checks.
pub fn get_current_memory() -> MemorySnapshot {
    MemorySnapshot::now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot() {
        let snapshot = MemorySnapshot::now();
        // Should have some memory allocated for the test framework
        assert!(snapshot.allocated > 0);
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();

        // Profile a simple allocation
        let result = profiler.profile("test_allocation", || {
            let v: Vec<u8> = vec![0u8; 1024 * 1024]; // 1MB
            v.len()
        });

        assert_eq!(result, 1024 * 1024);
        assert_eq!(profiler.profiles().len(), 1);

        let profile = &profiler.profiles()[0];
        assert_eq!(profile.operation, "test_allocation");
        // Peak should include the 1MB allocation
        assert!(profile.peak_bytes >= 1024 * 1024);
    }

    #[test]
    fn test_memory_estimate() {
        let estimate = MemoryEstimate::estimate(ModelSize::Tiny);
        assert!(estimate.param_count > 0);
        assert!(estimate.estimated_total_memory > 0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryProfile::format_bytes(500), "500 B");
        assert_eq!(MemoryProfile::format_bytes(1024), "1.0 KB");
        assert_eq!(MemoryProfile::format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(MemoryProfile::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
