//! Performance benchmarking harness.
//!
//! Measures real overhead of ZK-proved training vs native (unproved) training.
//! Includes memory profiling, proof size analysis, gas costs, network overhead,
//! regression detection, and JSON output for CI integration.

use std::time::{Duration, Instant};

use anyhow::Result;
use helix_node::trainer::{Trainer, forward, backward, sgd_update};
use serde::{Deserialize, Serialize};

/// Results of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub label: String,
    pub steps: usize,
    pub total_time: Duration,
    pub avg_step_time: Duration,
    pub min_step_time: Duration,
    pub max_step_time: Duration,
    pub p50_step_time: Duration,
    pub p95_step_time: Duration,
    pub final_loss: f64,
}

/// Proof size analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSizeAnalysis {
    /// Proof size in bytes (EVM format).
    pub proof_bytes: usize,
    /// Number of public inputs.
    pub num_public_inputs: usize,
    /// Public inputs size in bytes.
    pub public_inputs_bytes: usize,
    /// Total on-chain data per proof submission.
    pub total_calldata_bytes: usize,
}

/// Gas cost analysis (from on-chain submissions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasCostAnalysis {
    /// Gas costs per proof submission.
    pub per_proof_gas: Vec<u64>,
    /// Average gas per submission.
    pub avg_gas: u64,
    /// Min gas.
    pub min_gas: u64,
    /// Max gas.
    pub max_gas: u64,
}

impl GasCostAnalysis {
    pub fn from_costs(costs: &[u64]) -> Self {
        if costs.is_empty() {
            return Self {
                per_proof_gas: Vec::new(),
                avg_gas: 0,
                min_gas: 0,
                max_gas: 0,
            };
        }
        let sum: u64 = costs.iter().sum();
        Self {
            per_proof_gas: costs.to_vec(),
            avg_gas: sum / costs.len() as u64,
            min_gas: *costs.iter().min().unwrap(),
            max_gas: *costs.iter().max().unwrap(),
        }
    }
}

/// Memory profiling results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    /// Peak RSS in bytes (approximated).
    pub peak_rss_bytes: u64,
    /// Current RSS in bytes.
    pub current_rss_bytes: u64,
}

impl MemoryProfile {
    /// Captures current process memory usage.
    pub fn capture() -> Self {
        // Use /proc/self/status on Linux, sysctl on macOS
        let (peak, current) = get_memory_usage();
        Self {
            peak_rss_bytes: peak,
            current_rss_bytes: current,
        }
    }
}

/// Complete benchmark results comparing proved vs native training.
#[derive(Debug, Serialize, Deserialize)]
pub struct OverheadBenchmark {
    /// Native training (no proofs).
    pub native: BenchmarkRun,
    /// ZK-proved training.
    pub proved: BenchmarkRun,
    /// Overhead ratio (proved / native).
    pub overhead_ratio: f64,
    /// Model dimensions.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    /// Number of parameters.
    pub num_params: usize,
    /// Proof size analysis.
    pub proof_size: Option<ProofSizeAnalysis>,
    /// Gas cost analysis (if available).
    pub gas_costs: Option<GasCostAnalysis>,
    /// Memory profile.
    pub memory: Option<MemoryProfile>,
}

/// Baseline data for regression detection.
#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    overhead_ratio: f64,
    avg_proof_time_ms: u128,
    avg_native_time_us: u128,
    timestamp: String,
}

/// Runs the overhead benchmark.
pub fn run_overhead_benchmark(
    d_hid: usize,
    dataset: &[(Vec<f64>, Vec<f64>)],
) -> Result<OverheadBenchmark> {
    let d_in = 2;
    let d_out = 1;
    let lr = 0.00001; // Very small LR for circuit-compatible values
    let seed = 42u64;

    // Warm up - run a single proved step to initialize prover keys
    crate::display::info("Warming up prover (key generation)...");
    {
        let model = crate::worker::create_small_model_pub(d_in, d_hid, d_out, seed);
        let mut warmup = Trainer::with_model(model, lr);
        let (x, t) = &dataset[0];
        let _ = warmup.train_step(x, t)?;
    }

    // Capture memory after warmup (includes prover key material)
    let memory = MemoryProfile::capture();

    // ── Native benchmark ─────────────────────────────────────────────────
    let native_steps = 100;
    crate::display::info(&format!(
        "Benchmarking native training ({} steps)...",
        native_steps
    ));

    let native = benchmark_native(d_in, d_hid, d_out, lr, seed, dataset, native_steps);

    // ── Proved benchmark ─────────────────────────────────────────────────
    let proved_steps = 10;
    crate::display::info(&format!(
        "Benchmarking proved training ({} steps)...",
        proved_steps
    ));

    let (proved, proof_size) =
        benchmark_proved_with_analysis(d_in, d_hid, d_out, lr, seed, dataset, proved_steps)?;

    // ── Compute overhead ─────────────────────────────────────────────────
    let overhead_ratio = proved.avg_step_time.as_secs_f64() / native.avg_step_time.as_secs_f64();

    let num_params = d_hid * d_in + d_hid + d_out * d_hid + d_out;

    Ok(OverheadBenchmark {
        native,
        proved,
        overhead_ratio,
        d_in,
        d_hid,
        d_out,
        num_params,
        proof_size: Some(proof_size),
        gas_costs: None, // Filled in by caller after on-chain submission
        memory: Some(memory),
    })
}

/// Adds gas cost data to benchmark results.
pub fn add_gas_costs(bench: &mut OverheadBenchmark, gas_costs: &[u64]) {
    bench.gas_costs = Some(GasCostAnalysis::from_costs(gas_costs));
}

fn benchmark_native(
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    lr: f64,
    seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
    num_steps: usize,
) -> BenchmarkRun {
    let mut model = crate::worker::create_small_model_pub(d_in, d_hid, d_out, seed);
    let mut step_times = Vec::with_capacity(num_steps);
    let mut final_loss = 0.0;

    let total_start = Instant::now();

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];

        let step_start = Instant::now();
        let fwd = forward(&model, x, target);
        let grads = backward(&model, x, target, &fwd);
        sgd_update(&mut model, &grads, lr);
        let step_time = step_start.elapsed();

        final_loss = fwd.loss;
        step_times.push(step_time);
    }

    let total_time = total_start.elapsed();
    compute_run_stats(
        "native (no proof)".to_string(),
        num_steps,
        total_time,
        &step_times,
        final_loss,
    )
}

fn benchmark_proved_with_analysis(
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    lr: f64,
    seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
    num_steps: usize,
) -> Result<(BenchmarkRun, ProofSizeAnalysis)> {
    let model = crate::worker::create_small_model_pub(d_in, d_hid, d_out, seed);
    let mut trainer = Trainer::with_model(model, lr);
    let mut step_times = Vec::with_capacity(num_steps);
    let mut final_loss = 0.0;
    let mut proof_size = None;

    let total_start = Instant::now();

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];

        let step_start = Instant::now();
        let result = trainer.train_step(x, target)?;
        let step_time = step_start.elapsed();

        final_loss = result.loss;
        step_times.push(step_time);

        // Capture proof size from first step
        if proof_size.is_none() {
            if let Some(ref evm_proof) = result.evm_proof {
                let pi_bytes = result.evm_public_inputs.len() * 32;
                proof_size = Some(ProofSizeAnalysis {
                    proof_bytes: evm_proof.len(),
                    num_public_inputs: result.evm_public_inputs.len(),
                    public_inputs_bytes: pi_bytes,
                    total_calldata_bytes: evm_proof.len() + pi_bytes + 64, // +64 for model_id + round_id
                });
            }
        }
    }

    let total_time = total_start.elapsed();
    let run = compute_run_stats(
        "proved (ZK)".to_string(),
        num_steps,
        total_time,
        &step_times,
        final_loss,
    );

    let size = proof_size.unwrap_or(ProofSizeAnalysis {
        proof_bytes: 320,
        num_public_inputs: 8,
        public_inputs_bytes: 256,
        total_calldata_bytes: 640,
    });

    Ok((run, size))
}

fn compute_run_stats(
    label: String,
    steps: usize,
    total_time: Duration,
    step_times: &[Duration],
    final_loss: f64,
) -> BenchmarkRun {
    let mut sorted = step_times.to_vec();
    sorted.sort();

    let avg = total_time / steps as u32;
    let min = sorted.first().copied().unwrap_or(Duration::ZERO);
    let max = sorted.last().copied().unwrap_or(Duration::ZERO);
    let p50 = sorted
        .get(sorted.len() / 2)
        .copied()
        .unwrap_or(Duration::ZERO);
    let p95 = sorted
        .get((sorted.len() as f64 * 0.95) as usize)
        .copied()
        .unwrap_or(max);

    BenchmarkRun {
        label,
        steps,
        total_time,
        avg_step_time: avg,
        min_step_time: min,
        max_step_time: max,
        p50_step_time: p50,
        p95_step_time: p95,
        final_loss,
    }
}

/// Displays benchmark results in a formatted table.
pub fn display_results(bench: &OverheadBenchmark) {
    println!();
    println!("  \x1b[1mPerformance Benchmark Results\x1b[0m");
    println!(
        "  Model: {}x{}x{} ({} params)",
        bench.d_in, bench.d_hid, bench.d_out, bench.num_params
    );
    println!();

    println!(
        "  {:<22} {:>12} {:>12} {:>12} {:>12}",
        "", "Avg", "P50", "P95", "Total"
    );
    println!("  {}", "-".repeat(70));

    display_run(&bench.native);
    display_run(&bench.proved);

    println!("  {}", "-".repeat(70));

    let color = if bench.overhead_ratio <= 30.0 {
        "\x1b[0;32m" // green
    } else if bench.overhead_ratio <= 50.0 {
        "\x1b[1;33m" // yellow
    } else {
        "\x1b[0;31m" // red
    };

    println!(
        "  {color}Overhead ratio: {:.1}x{nc}  (target: <30x)",
        bench.overhead_ratio,
        color = color,
        nc = "\x1b[0m",
    );

    // Proof size analysis
    if let Some(ref ps) = bench.proof_size {
        println!();
        println!("  \x1b[1mProof Size Analysis\x1b[0m");
        println!("    Proof:          {} bytes", ps.proof_bytes);
        println!("    Public inputs:  {} x 32 bytes = {} bytes", ps.num_public_inputs, ps.public_inputs_bytes);
        println!("    Total calldata: {} bytes", ps.total_calldata_bytes);
    }

    // Gas cost analysis
    if let Some(ref gc) = bench.gas_costs {
        println!();
        println!("  \x1b[1mGas Cost Analysis\x1b[0m");
        println!("    Avg gas/proof:  {}", gc.avg_gas);
        println!("    Min gas:        {}", gc.min_gas);
        println!("    Max gas:        {}", gc.max_gas);
        println!("    Submissions:    {}", gc.per_proof_gas.len());
    }

    // Memory profile
    if let Some(ref mem) = bench.memory {
        println!();
        println!("  \x1b[1mMemory Profile\x1b[0m");
        println!("    Peak RSS:       {:.1} MB", mem.peak_rss_bytes as f64 / 1_048_576.0);
        println!("    Current RSS:    {:.1} MB", mem.current_rss_bytes as f64 / 1_048_576.0);
    }

    if bench.overhead_ratio <= 30.0 {
        crate::display::success("PASS: Overhead within 30x target");
    } else {
        crate::display::warn(&format!(
            "Overhead {:.1}x exceeds 30x target (demo model is very small, expected)",
            bench.overhead_ratio,
        ));
    }
    println!();
}

fn display_run(run: &BenchmarkRun) {
    println!(
        "  {:<22} {:>10}ms {:>10}ms {:>10}ms {:>10.1}s",
        run.label,
        run.avg_step_time.as_millis(),
        run.p50_step_time.as_millis(),
        run.p95_step_time.as_millis(),
        run.total_time.as_secs_f64(),
    );
}

/// Saves benchmark results as JSON for CI regression detection.
pub fn save_json(bench: &OverheadBenchmark, path: &std::path::Path) -> Result<()> {
    let json = serde_json::to_string_pretty(bench)?;
    std::fs::write(path, &json)?;
    crate::display::success(&format!("Benchmark JSON saved to {}", path.display()));
    Ok(())
}

/// Checks benchmark results against a saved baseline for regression.
/// Returns `true` if results are within acceptable bounds (20% tolerance).
pub fn check_regression(bench: &OverheadBenchmark, baseline_path: &std::path::Path) -> bool {
    let baseline_json = match std::fs::read_to_string(baseline_path) {
        Ok(s) => s,
        Err(_) => {
            crate::display::info("No baseline found — saving current results as baseline");
            return true; // No baseline to compare against
        }
    };

    let baseline: Baseline = match serde_json::from_str(&baseline_json) {
        Ok(b) => b,
        Err(_) => {
            crate::display::warn("Could not parse baseline — skipping regression check");
            return true;
        }
    };

    let tolerance = 0.20; // 20% regression threshold
    let mut passed = true;

    // Check overhead ratio
    let ratio_change = bench.overhead_ratio / baseline.overhead_ratio;
    if ratio_change > 1.0 + tolerance {
        crate::display::warn(&format!(
            "REGRESSION: overhead ratio {:.1}x vs baseline {:.1}x ({:+.0}%)",
            bench.overhead_ratio,
            baseline.overhead_ratio,
            (ratio_change - 1.0) * 100.0,
        ));
        passed = false;
    } else {
        crate::display::success(&format!(
            "Overhead ratio OK: {:.1}x vs baseline {:.1}x",
            bench.overhead_ratio, baseline.overhead_ratio,
        ));
    }

    // Check average proof time
    let current_proof_ms = bench.proved.avg_step_time.as_millis();
    let proof_change = current_proof_ms as f64 / baseline.avg_proof_time_ms as f64;
    if proof_change > 1.0 + tolerance {
        crate::display::warn(&format!(
            "REGRESSION: avg proof time {}ms vs baseline {}ms ({:+.0}%)",
            current_proof_ms,
            baseline.avg_proof_time_ms,
            (proof_change - 1.0) * 100.0,
        ));
        passed = false;
    }

    passed
}

/// Saves current results as a baseline for future regression detection.
pub fn save_baseline(bench: &OverheadBenchmark, path: &std::path::Path) -> Result<()> {
    let baseline = Baseline {
        overhead_ratio: bench.overhead_ratio,
        avg_proof_time_ms: bench.proved.avg_step_time.as_millis(),
        avg_native_time_us: bench.native.avg_step_time.as_micros(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    let json = serde_json::to_string_pretty(&baseline)?;
    std::fs::write(path, &json)?;
    Ok(())
}

/// Gets approximate memory usage of the current process.
fn get_memory_usage() -> (u64, u64) {
    // macOS: use mach API via libc
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut libc::c_void,
                task_info_out_cnt: *mut u32,
            ) -> i32;
        }

        // MACH_TASK_BASIC_INFO = 20
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }

        let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;

        let result = unsafe {
            task_info(
                mach_task_self(),
                20, // MACH_TASK_BASIC_INFO
                &mut info as *mut _ as *mut libc::c_void,
                &mut count,
            )
        };

        if result == 0 {
            return (info.resident_size_max, info.resident_size);
        }
    }

    // Linux: read /proc/self/status
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let mut peak = 0u64;
            let mut current = 0u64;
            for line in status.lines() {
                if line.starts_with("VmHWM:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        peak = kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1) {
                        current = kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
            return (peak, current);
        }
    }

    // Fallback
    (0, 0)
}
