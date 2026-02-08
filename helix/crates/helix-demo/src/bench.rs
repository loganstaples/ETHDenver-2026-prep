//! Performance benchmarking harness.
//!
//! Measures real overhead of ZK-proved training vs native (unproved) training.
//! Target: <30x overhead for the demo model configuration.

use std::time::{Duration, Instant};

use anyhow::Result;
use helix_node::trainer::{Trainer, forward, backward, sgd_update};

/// Results of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    pub label: String,
    #[allow(dead_code)]
    pub steps: usize,
    pub total_time: Duration,
    pub avg_step_time: Duration,
    #[allow(dead_code)]
    pub min_step_time: Duration,
    #[allow(dead_code)]
    pub max_step_time: Duration,
    pub p50_step_time: Duration,
    pub p95_step_time: Duration,
    #[allow(dead_code)]
    pub final_loss: f64,
}

/// Complete benchmark results comparing proved vs native training.
#[derive(Debug)]
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
}

/// Runs the overhead benchmark.
///
/// Measures native training time vs proved training time and computes the ratio.
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

    let proved = benchmark_proved(d_in, d_hid, d_out, lr, seed, dataset, proved_steps)?;

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
    })
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
    compute_run_stats("native (no proof)".to_string(), num_steps, total_time, &step_times, final_loss)
}

fn benchmark_proved(
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    lr: f64,
    seed: u64,
    dataset: &[(Vec<f64>, Vec<f64>)],
    num_steps: usize,
) -> Result<BenchmarkRun> {
    let model = crate::worker::create_small_model_pub(d_in, d_hid, d_out, seed);
    let mut trainer = Trainer::with_model(model, lr);
    let mut step_times = Vec::with_capacity(num_steps);
    let mut final_loss = 0.0;

    let total_start = Instant::now();

    for step in 0..num_steps {
        let (x, target) = &dataset[step % dataset.len()];

        let step_start = Instant::now();
        let result = trainer.train_step(x, target)?;
        let step_time = step_start.elapsed();

        final_loss = result.loss;
        step_times.push(step_time);
    }

    let total_time = total_start.elapsed();
    Ok(compute_run_stats(
        "proved (ZK)".to_string(),
        num_steps,
        total_time,
        &step_times,
        final_loss,
    ))
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
    let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(Duration::ZERO);
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
    println!(
        "  \x1b[1mPerformance Benchmark Results\x1b[0m"
    );
    println!(
        "  Model: {}x{}x{} ({} params)",
        bench.d_in, bench.d_hid, bench.d_out, bench.num_params
    );
    println!();

    println!("  {:<22} {:>12} {:>12} {:>12} {:>12}", "", "Avg", "P50", "P95", "Total");
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
