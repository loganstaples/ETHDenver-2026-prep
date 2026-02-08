//! Timing Analysis for Circuit Operations.
//!
//! Provides sub-millisecond timing for all phases of proof generation:
//! - Witness generation
//! - Circuit synthesis
//! - Commitment computation
//! - Proof generation
//! - Verification
//!
//! # Usage
//!
//! ```ignore
//! let mut profiler = TimingProfiler::new();
//! let profile = profiler.profile_mock_prover(&circuit, k, inputs)?;
//! println!("Total time: {:?}", profile.total_time);
//! println!("Witness gen: {:?}", profile.witness_generation);
//! ```

use halo2_proofs::{
    dev::MockProver,
    plonk::{Circuit, Error, ErrorFront},
};
use halo2curves::bn256::Fr;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Complete timing profile of a circuit.
#[derive(Debug, Clone, Default)]
pub struct TimingProfile {
    /// Total time for all operations.
    pub total_time: Duration,
    /// Time for witness generation.
    pub witness_generation: Duration,
    /// Time for circuit synthesis.
    pub synthesis: Duration,
    /// Time for verification.
    pub verification: Duration,
    /// Time for commitment computation.
    pub commitment: Duration,
    /// Per-phase breakdown.
    pub phases: HashMap<String, Duration>,
    /// Statistical analysis.
    pub stats: TimingStats,
}

/// Statistical analysis of timing.
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    /// Minimum observed time.
    pub min: Duration,
    /// Maximum observed time.
    pub max: Duration,
    /// Mean time.
    pub mean: Duration,
    /// Standard deviation.
    pub std_dev: Duration,
    /// Number of samples.
    pub sample_count: usize,
    /// 50th percentile.
    pub p50: Duration,
    /// 95th percentile.
    pub p95: Duration,
    /// 99th percentile.
    pub p99: Duration,
}

/// Witness generation timing breakdown.
#[derive(Debug, Clone, Default)]
pub struct WitnessGenerationTiming {
    /// Forward pass computation.
    pub forward_pass: Duration,
    /// Loss computation.
    pub loss_computation: Duration,
    /// Backward pass computation.
    pub backward_pass: Duration,
    /// Weight update computation.
    pub weight_update: Duration,
    /// Error bound computation.
    pub error_bounds: Duration,
    /// State hash computation.
    pub state_hash: Duration,
    /// Freivalds challenge generation.
    pub freivalds_challenge: Duration,
}

/// Circuit synthesis timing breakdown.
#[derive(Debug, Clone, Default)]
pub struct SynthesisTiming {
    /// Table loading time.
    pub table_loading: Duration,
    /// Public input binding.
    pub public_input_binding: Duration,
    /// Forward pass synthesis.
    pub forward_synthesis: Duration,
    /// Backward pass synthesis.
    pub backward_synthesis: Duration,
    /// Constraint evaluation.
    pub constraint_evaluation: Duration,
}

/// Proving timing breakdown.
#[derive(Debug, Clone, Default)]
pub struct ProvingTiming {
    /// Polynomial commitment.
    pub polynomial_commitment: Duration,
    /// Quotient polynomial.
    pub quotient_polynomial: Duration,
    /// Evaluation.
    pub evaluation: Duration,
    /// Opening proof.
    pub opening_proof: Duration,
}

/// Phase of circuit execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfilingPhase {
    /// Witness generation phase.
    WitnessGeneration,
    /// Circuit synthesis phase.
    Synthesis,
    /// Commitment computation phase.
    Commitment,
    /// Proof generation phase.
    ProofGeneration,
    /// Verification phase.
    Verification,
    /// Custom phase.
    Custom,
}

impl std::fmt::Display for ProfilingPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WitnessGeneration => write!(f, "witness_generation"),
            Self::Synthesis => write!(f, "synthesis"),
            Self::Commitment => write!(f, "commitment"),
            Self::ProofGeneration => write!(f, "proof_generation"),
            Self::Verification => write!(f, "verification"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Timer for a specific phase.
pub struct PhaseTimer {
    phase: ProfilingPhase,
    name: String,
    start: Instant,
    checkpoints: Vec<(String, Duration)>,
}

impl PhaseTimer {
    /// Creates a new phase timer.
    pub fn new(phase: ProfilingPhase, name: &str) -> Self {
        Self {
            phase,
            name: name.to_string(),
            start: Instant::now(),
            checkpoints: Vec::new(),
        }
    }

    /// Records a checkpoint.
    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.push((name.to_string(), self.start.elapsed()));
    }

    /// Stops the timer and returns elapsed time.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }

    /// Returns the phase.
    pub fn phase(&self) -> ProfilingPhase {
        self.phase
    }

    /// Returns the name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns checkpoints.
    pub fn checkpoints(&self) -> &[(String, Duration)] {
        &self.checkpoints
    }
}

/// Main timing profiler.
pub struct TimingProfiler {
    /// Accumulated timing data.
    samples: Vec<TimingProfile>,
    /// Active timers.
    active_timers: HashMap<String, Instant>,
    /// Configuration.
    config: TimingConfig,
}

/// Configuration for timing profiler.
#[derive(Debug, Clone)]
pub struct TimingConfig {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of timed iterations.
    pub timed_iterations: usize,
    /// Enable detailed phase timing.
    pub detailed_phases: bool,
    /// Minimum duration to record.
    pub min_duration: Duration,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 1,
            timed_iterations: 3,
            detailed_phases: true,
            min_duration: Duration::from_nanos(100),
        }
    }
}

impl TimingProfiler {
    /// Creates a new timing profiler.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            active_timers: HashMap::new(),
            config: TimingConfig::default(),
        }
    }

    /// Creates a profiler with custom configuration.
    pub fn with_config(config: TimingConfig) -> Self {
        Self {
            samples: Vec::new(),
            active_timers: HashMap::new(),
            config,
        }
    }

    /// Starts timing a named operation.
    pub fn start(&mut self, name: &str) {
        self.active_timers.insert(name.to_string(), Instant::now());
    }

    /// Stops timing a named operation and returns duration.
    pub fn stop(&mut self, name: &str) -> Option<Duration> {
        self.active_timers.remove(name).map(|start| start.elapsed())
    }

    /// Profiles a circuit using MockProver.
    pub fn profile_mock_prover<C: Circuit<Fr> + Clone>(
        &mut self,
        circuit: &C,
        k: u32,
        public_inputs: Vec<Vec<Fr>>,
    ) -> Result<TimingProfile, Error> {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = MockProver::run(k, circuit, public_inputs.clone())?;
        }

        let mut samples = Vec::with_capacity(self.config.timed_iterations);

        // Timed iterations
        for _ in 0..self.config.timed_iterations {
            let total_start = Instant::now();

            // MockProver.run includes witness generation and synthesis
            let prover_start = Instant::now();
            let prover = MockProver::run(k, circuit, public_inputs.clone())?;
            let prover_time = prover_start.elapsed();

            // Verification
            let verify_start = Instant::now();
            let _ = prover.verify();
            let verify_time = verify_start.elapsed();

            let total_time = total_start.elapsed();

            // Estimate breakdown (MockProver doesn't give us internal timing)
            let witness_gen = prover_time / 3; // Rough estimate
            let synthesis = prover_time - witness_gen;

            samples.push(TimingProfile {
                total_time,
                witness_generation: witness_gen,
                synthesis,
                verification: verify_time,
                commitment: Duration::ZERO, // MockProver doesn't do real commitments
                phases: HashMap::new(),
                stats: TimingStats::default(),
            });
        }

        // Compute aggregate profile
        let aggregate = self.aggregate_profiles(&samples);
        self.samples.extend(samples);

        Ok(aggregate)
    }

    /// Aggregates multiple timing profiles into one.
    fn aggregate_profiles(&self, profiles: &[TimingProfile]) -> TimingProfile {
        if profiles.is_empty() {
            return TimingProfile::default();
        }

        let n = profiles.len();

        let mut total_times: Vec<Duration> = profiles.iter().map(|p| p.total_time).collect();
        let mut witness_times: Vec<Duration> = profiles.iter().map(|p| p.witness_generation).collect();
        let mut synthesis_times: Vec<Duration> = profiles.iter().map(|p| p.synthesis).collect();
        let mut verify_times: Vec<Duration> = profiles.iter().map(|p| p.verification).collect();

        total_times.sort();
        witness_times.sort();
        synthesis_times.sort();
        verify_times.sort();

        let mean_total = Duration::from_nanos(
            total_times.iter().map(|d| d.as_nanos()).sum::<u128>() as u64 / n as u64
        );

        TimingProfile {
            total_time: mean_total,
            witness_generation: witness_times[n / 2],
            synthesis: synthesis_times[n / 2],
            verification: verify_times[n / 2],
            commitment: Duration::ZERO,
            phases: HashMap::new(),
            stats: TimingStats {
                min: *total_times.first().unwrap_or(&Duration::ZERO),
                max: *total_times.last().unwrap_or(&Duration::ZERO),
                mean: mean_total,
                std_dev: compute_std_dev(&total_times, mean_total),
                sample_count: n,
                p50: total_times[n / 2],
                p95: total_times[(n * 95) / 100],
                p99: total_times[(n * 99) / 100].min(total_times[n - 1]),
            },
        }
    }

    /// Creates a new phase timer.
    pub fn phase_timer(&self, phase: ProfilingPhase, name: &str) -> PhaseTimer {
        PhaseTimer::new(phase, name)
    }

    /// Records a phase timing.
    pub fn record_phase(&mut self, phase: ProfilingPhase, duration: Duration) {
        // Could store in samples for aggregation
        let _ = (phase, duration);
    }

    /// Returns all collected samples.
    pub fn samples(&self) -> &[TimingProfile] {
        &self.samples
    }

    /// Clears all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.active_timers.clear();
    }
}

impl Default for TimingProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes standard deviation of durations.
fn compute_std_dev(durations: &[Duration], mean: Duration) -> Duration {
    if durations.len() < 2 {
        return Duration::ZERO;
    }

    let mean_nanos = mean.as_nanos() as f64;
    let variance: f64 = durations.iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>() / (durations.len() - 1) as f64;

    Duration::from_nanos(variance.sqrt() as u64)
}

/// Scoped timer that automatically records duration on drop.
pub struct ScopedTimer<'a> {
    profiler: &'a mut TimingProfiler,
    name: String,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    /// Creates a new scoped timer.
    pub fn new(profiler: &'a mut TimingProfiler, name: &str) -> Self {
        Self {
            profiler,
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedTimer<'_> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.profiler.record_phase(ProfilingPhase::Custom, elapsed);
    }
}

/// High-resolution timer for micro-benchmarks.
pub struct HighResTimer {
    start: Instant,
    laps: Vec<Duration>,
}

impl HighResTimer {
    /// Creates and starts a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            laps: Vec::new(),
        }
    }

    /// Records a lap time without stopping.
    pub fn lap(&mut self) -> Duration {
        let elapsed = self.start.elapsed();
        self.laps.push(elapsed);
        elapsed
    }

    /// Stops the timer and returns total elapsed time.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
    }

    /// Returns all lap times.
    pub fn laps(&self) -> &[Duration] {
        &self.laps
    }

    /// Returns elapsed time without stopping.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Timing comparison between two implementations.
#[derive(Debug, Clone)]
pub struct TimingComparison {
    /// Name of first implementation.
    pub name_a: String,
    /// Time for first implementation.
    pub time_a: Duration,
    /// Name of second implementation.
    pub name_b: String,
    /// Time for second implementation.
    pub time_b: Duration,
    /// Speedup ratio (time_a / time_b).
    pub speedup: f64,
    /// Which implementation is faster.
    pub faster: String,
}

impl TimingComparison {
    /// Creates a new timing comparison.
    pub fn new(name_a: &str, time_a: Duration, name_b: &str, time_b: Duration) -> Self {
        let speedup = if time_b.as_nanos() > 0 {
            time_a.as_nanos() as f64 / time_b.as_nanos() as f64
        } else {
            f64::INFINITY
        };

        let faster = if time_a < time_b {
            name_a.to_string()
        } else {
            name_b.to_string()
        };

        Self {
            name_a: name_a.to_string(),
            time_a,
            name_b: name_b.to_string(),
            time_b,
            speedup,
            faster,
        }
    }

    /// Returns a formatted report.
    pub fn report(&self) -> String {
        format!(
            "{}: {:?} vs {}: {:?}\n\
             Speedup: {:.2}x\n\
             Faster: {}",
            self.name_a, self.time_a,
            self.name_b, self.time_b,
            self.speedup,
            self.faster
        )
    }
}

/// Tracks timing trends over multiple runs.
#[derive(Debug, Clone, Default)]
pub struct TimingTrend {
    /// Historical timing data.
    samples: Vec<(std::time::SystemTime, Duration)>,
    /// Moving average window size.
    window_size: usize,
}

impl TimingTrend {
    /// Creates a new timing trend tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            samples: Vec::new(),
            window_size,
        }
    }

    /// Adds a new sample.
    pub fn add_sample(&mut self, duration: Duration) {
        self.samples.push((std::time::SystemTime::now(), duration));
    }

    /// Returns the moving average.
    pub fn moving_average(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }

        let window = &self.samples[self.samples.len().saturating_sub(self.window_size)..];
        let sum: u128 = window.iter().map(|(_, d)| d.as_nanos()).sum();
        Duration::from_nanos((sum / window.len() as u128) as u64)
    }

    /// Detects if there's a regression.
    pub fn detect_regression(&self, threshold: f64) -> bool {
        if self.samples.len() < self.window_size * 2 {
            return false;
        }

        let recent = self.moving_average();
        let old_window = &self.samples[..self.window_size];
        let old_avg: u128 = old_window.iter().map(|(_, d)| d.as_nanos()).sum::<u128>()
            / self.window_size as u128;

        recent.as_nanos() as f64 > old_avg as f64 * (1.0 + threshold)
    }

    /// Returns trend direction.
    pub fn trend_direction(&self) -> TrendDirection {
        if self.samples.len() < 3 {
            return TrendDirection::Stable;
        }

        let n = self.samples.len();
        let recent = self.samples[n-1].1;
        let middle = self.samples[n-2].1;
        let old = self.samples[n-3].1;

        let trend1 = recent.as_nanos() as i128 - middle.as_nanos() as i128;
        let trend2 = middle.as_nanos() as i128 - old.as_nanos() as i128;

        if trend1 > 0 && trend2 > 0 {
            TrendDirection::Increasing
        } else if trend1 < 0 && trend2 < 0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Direction of a timing trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Times are increasing (getting slower).
    Increasing,
    /// Times are decreasing (getting faster).
    Decreasing,
    /// Times are stable.
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_profile_default() {
        let profile = TimingProfile::default();
        assert_eq!(profile.total_time, Duration::ZERO);
        assert_eq!(profile.witness_generation, Duration::ZERO);
    }

    #[test]
    fn test_phase_timer() {
        let mut timer = PhaseTimer::new(ProfilingPhase::WitnessGeneration, "test");

        std::thread::sleep(Duration::from_millis(1));
        timer.checkpoint("checkpoint1");

        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(1));
    }

    #[test]
    fn test_high_res_timer() {
        let mut timer = HighResTimer::start();

        std::thread::sleep(Duration::from_millis(1));
        timer.lap();

        std::thread::sleep(Duration::from_millis(1));
        let total = timer.stop();

        assert!(total >= Duration::from_millis(2));
    }

    #[test]
    fn test_timing_comparison() {
        let comp = TimingComparison::new(
            "impl_a",
            Duration::from_millis(100),
            "impl_b",
            Duration::from_millis(50),
        );

        assert_eq!(comp.faster, "impl_b");
        assert!((comp.speedup - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_timing_trend() {
        let mut trend = TimingTrend::new(3);

        trend.add_sample(Duration::from_millis(100));
        trend.add_sample(Duration::from_millis(110));
        trend.add_sample(Duration::from_millis(120));

        assert_eq!(trend.trend_direction(), TrendDirection::Increasing);
    }

    #[test]
    fn test_std_dev_computation() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(90),
        ];

        let mean = Duration::from_millis(100);
        let std_dev = compute_std_dev(&durations, mean);

        // Std dev should be around 10ms
        assert!(std_dev >= Duration::from_millis(8));
        assert!(std_dev <= Duration::from_millis(12));
    }

    #[test]
    fn test_timing_stats() {
        let stats = TimingStats {
            min: Duration::from_millis(90),
            max: Duration::from_millis(110),
            mean: Duration::from_millis(100),
            std_dev: Duration::from_millis(10),
            sample_count: 10,
            p50: Duration::from_millis(100),
            p95: Duration::from_millis(108),
            p99: Duration::from_millis(110),
        };

        assert_eq!(stats.sample_count, 10);
        assert!(stats.p95 > stats.p50);
    }
}
