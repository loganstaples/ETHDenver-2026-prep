//! Demo Orchestrator - Precise Timing Control
//!
//! Provides precise timing control for 90-second demo execution with:
//! - Phase scheduling with timing guarantees
//! - Adaptive pacing based on actual performance
//! - Graceful degradation under load
//! - Event synchronization across components

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock, watch};

use crate::rpc::{
    HelixRpcConfig, MockRpcClient, ProofPhase, ProofStatus, RealTimeProofTracker,
    RealTimeTrainingTracker, TrainingPhase, TrainingProgress, TrainingStatus,
    UnifiedRpcClient, WorkerInfo,
};

/// Maximum demo duration (90 seconds)
pub const MAX_DEMO_DURATION: Duration = Duration::from_secs(90);

/// Target demo duration (85 seconds to leave buffer)
pub const TARGET_DEMO_DURATION: Duration = Duration::from_secs(85);

/// Minimum time per phase (ensures visibility)
pub const MIN_PHASE_DURATION: Duration = Duration::from_millis(500);

/// Demo timing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoTimingConfig {
    /// Total demo duration target
    pub total_duration: Duration,
    /// Hard deadline (must complete by this time)
    pub hard_deadline: Duration,
    /// Time allocated for setup
    pub setup_budget: Duration,
    /// Time allocated per training round
    pub round_budget: Duration,
    /// Time allocated for proof generation display
    pub proof_display_budget: Duration,
    /// Time allocated for finalization
    pub finalization_budget: Duration,
    /// Buffer time for overruns
    pub buffer_time: Duration,
    /// Minimum time to show each phase
    pub min_phase_visibility: Duration,
    /// Enable adaptive pacing
    pub adaptive_pacing: bool,
}

impl Default for DemoTimingConfig {
    fn default() -> Self {
        Self::for_duration(TARGET_DEMO_DURATION, 5)
    }
}

impl DemoTimingConfig {
    /// Create timing config for specified duration and round count
    pub fn for_duration(duration: Duration, rounds: u32) -> Self {
        let total_secs = duration.as_secs_f64();

        // Budget allocation:
        // - Setup: 10%
        // - Training rounds: 70%
        // - Finalization: 10%
        // - Buffer: 10%

        let setup_secs = total_secs * 0.10;
        let training_secs = total_secs * 0.70;
        let finalization_secs = total_secs * 0.10;
        let buffer_secs = total_secs * 0.10;

        let round_secs = training_secs / rounds as f64;

        Self {
            total_duration: duration,
            hard_deadline: Duration::from_secs((total_secs + buffer_secs) as u64),
            setup_budget: Duration::from_secs_f64(setup_secs),
            round_budget: Duration::from_secs_f64(round_secs),
            proof_display_budget: Duration::from_secs_f64(round_secs * 0.3),
            finalization_budget: Duration::from_secs_f64(finalization_secs),
            buffer_time: Duration::from_secs_f64(buffer_secs),
            min_phase_visibility: MIN_PHASE_DURATION,
            adaptive_pacing: true,
        }
    }

    /// Create timing for 90-second demo
    pub fn standard_90_second(rounds: u32) -> Self {
        Self::for_duration(TARGET_DEMO_DURATION, rounds)
    }

    /// Create timing for quick demo (30 seconds)
    pub fn quick_30_second(rounds: u32) -> Self {
        Self::for_duration(Duration::from_secs(30), rounds)
    }
}

/// Demo phase with timing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrchestratedPhase {
    /// Pre-start preparation
    PreStart,
    /// Initializing network
    Initializing,
    /// Deploying contracts
    DeployingContracts,
    /// Registering model
    RegisteringModel,
    /// Starting worker nodes
    StartingWorkers,
    /// Staking tokens
    Staking,
    /// Training round in progress
    TrainingRound(u32),
    /// Generating proof
    GeneratingProof(u32),
    /// Verifying proof
    VerifyingProof(u32),
    /// Slashing event (for slashing demo)
    SlashingEvent,
    /// Finalizing model
    Finalizing,
    /// Demo complete
    Complete,
    /// Demo failed
    Failed,
}

impl OrchestratedPhase {
    pub fn name(&self) -> String {
        match self {
            Self::PreStart => "Pre-Start".to_string(),
            Self::Initializing => "Initializing".to_string(),
            Self::DeployingContracts => "Deploying Contracts".to_string(),
            Self::RegisteringModel => "Registering Model".to_string(),
            Self::StartingWorkers => "Starting Workers".to_string(),
            Self::Staking => "Staking".to_string(),
            Self::TrainingRound(r) => format!("Training Round {}", r),
            Self::GeneratingProof(r) => format!("Generating Proof {}", r),
            Self::VerifyingProof(r) => format!("Verifying Proof {}", r),
            Self::SlashingEvent => "Slashing Event".to_string(),
            Self::Finalizing => "Finalizing".to_string(),
            Self::Complete => "Complete".to_string(),
            Self::Failed => "Failed".to_string(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Complete | Self::Failed)
    }
}

/// Phase timing result
#[derive(Debug, Clone)]
pub struct PhaseTiming {
    /// Phase that was executed
    pub phase: OrchestratedPhase,
    /// Planned duration
    pub planned_duration: Duration,
    /// Actual duration
    pub actual_duration: Duration,
    /// Whether phase was within budget
    pub within_budget: bool,
    /// Overrun or underrun time
    pub variance: i64,
}

/// Orchestrator state snapshot
#[derive(Debug, Clone)]
pub struct OrchestratorSnapshot {
    /// Current phase
    pub phase: OrchestratedPhase,
    /// Total elapsed time
    pub elapsed: Duration,
    /// Time remaining
    pub remaining: Duration,
    /// Whether on schedule
    pub on_schedule: bool,
    /// Current pace factor (1.0 = on time, >1.0 = behind, <1.0 = ahead)
    pub pace_factor: f64,
    /// Completed phases
    pub completed_phases: Vec<PhaseTiming>,
    /// Training progress
    pub training_progress: Option<TrainingStatus>,
    /// Proof progress
    pub proof_progress: Option<ProofStatus>,
    /// Worker states
    pub workers: Vec<WorkerInfo>,
}

/// Event emitted by orchestrator
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Phase started
    PhaseStarted {
        phase: OrchestratedPhase,
        budget: Duration,
    },
    /// Phase completed
    PhaseCompleted {
        phase: OrchestratedPhase,
        timing: PhaseTiming,
    },
    /// Training round completed
    RoundCompleted {
        round: u32,
        loss: f64,
        error_bound: f64,
    },
    /// Proof generated
    ProofGenerated {
        round: u32,
        time_ms: u64,
    },
    /// Proof verified
    ProofVerified {
        round: u32,
        success: bool,
    },
    /// Worker slashed
    WorkerSlashed {
        worker_id: String,
        reason: String,
    },
    /// Timing warning
    TimingWarning {
        message: String,
        remaining: Duration,
    },
    /// Demo completed
    DemoCompleted {
        success: bool,
        total_time: Duration,
    },
    /// Error occurred
    Error {
        message: String,
        recoverable: bool,
    },
}

/// Demo Orchestrator with precise timing control
pub struct DemoOrchestrator {
    /// Timing configuration
    timing: DemoTimingConfig,
    /// Number of training rounds
    round_count: u32,
    /// Number of workers
    worker_count: u32,
    /// Whether to simulate slashing
    simulate_slashing: bool,
    /// Round at which to trigger slashing
    slashing_round: Option<u32>,
    /// Unified RPC client (real or mock)
    rpc_client: Arc<RwLock<UnifiedRpcClient>>,
    /// Real-time proof tracker
    proof_tracker: Arc<RealTimeProofTracker>,
    /// Real-time training tracker
    training_tracker: Arc<RealTimeTrainingTracker>,
    /// Current phase
    current_phase: Arc<RwLock<OrchestratedPhase>>,
    /// Demo start time
    start_time: Arc<RwLock<Option<Instant>>>,
    /// Phase timings
    phase_timings: Arc<RwLock<Vec<PhaseTiming>>>,
    /// Event sender
    event_tx: mpsc::Sender<OrchestratorEvent>,
    /// Shutdown receiver
    shutdown_rx: watch::Receiver<bool>,
}

impl DemoOrchestrator {
    /// Create new orchestrator
    pub fn new(
        timing: DemoTimingConfig,
        round_count: u32,
        worker_count: u32,
        event_tx: mpsc::Sender<OrchestratorEvent>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            timing,
            round_count,
            worker_count,
            simulate_slashing: false,
            slashing_round: None,
            rpc_client: Arc::new(RwLock::new(UnifiedRpcClient::mock_only())),
            proof_tracker: Arc::new(RealTimeProofTracker::new()),
            training_tracker: Arc::new(RealTimeTrainingTracker::new()),
            current_phase: Arc::new(RwLock::new(OrchestratedPhase::PreStart)),
            start_time: Arc::new(RwLock::new(None)),
            phase_timings: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            shutdown_rx,
        }
    }

    /// Create orchestrator with real node connection
    pub async fn with_endpoint(
        timing: DemoTimingConfig,
        round_count: u32,
        worker_count: u32,
        endpoint: &str,
        event_tx: mpsc::Sender<OrchestratorEvent>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let config = HelixRpcConfig::with_endpoint(endpoint);
        let client = UnifiedRpcClient::new(config).await;

        Self {
            timing,
            round_count,
            worker_count,
            simulate_slashing: false,
            slashing_round: None,
            rpc_client: Arc::new(RwLock::new(client)),
            proof_tracker: Arc::new(RealTimeProofTracker::new()),
            training_tracker: Arc::new(RealTimeTrainingTracker::new()),
            current_phase: Arc::new(RwLock::new(OrchestratedPhase::PreStart)),
            start_time: Arc::new(RwLock::new(None)),
            phase_timings: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            shutdown_rx,
        }
    }

    /// Check if connected to real node
    pub async fn is_real_mode(&self) -> bool {
        self.rpc_client.read().await.is_connected()
    }

    /// Create orchestrator for slashing demo
    pub fn with_slashing(mut self, slashing_round: u32) -> Self {
        self.simulate_slashing = true;
        self.slashing_round = Some(slashing_round);
        self
    }

    /// Get current phase
    pub async fn current_phase(&self) -> OrchestratedPhase {
        *self.current_phase.read().await
    }

    /// Get elapsed time
    pub async fn elapsed(&self) -> Duration {
        self.start_time
            .read()
            .await
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Get remaining time
    pub async fn remaining(&self) -> Duration {
        let elapsed = self.elapsed().await;
        self.timing.hard_deadline.saturating_sub(elapsed)
    }

    /// Check if demo should continue
    async fn should_continue(&self) -> bool {
        if *self.shutdown_rx.borrow() {
            return false;
        }

        let elapsed = self.elapsed().await;
        elapsed < self.timing.hard_deadline
    }

    /// Get current pace factor
    async fn pace_factor(&self) -> f64 {
        let elapsed = self.elapsed().await;
        let target_progress = elapsed.as_secs_f64() / self.timing.total_duration.as_secs_f64();

        let actual_progress = {
            let timings = self.phase_timings.read().await;
            let rounds_done = timings
                .iter()
                .filter(|t| matches!(t.phase, OrchestratedPhase::TrainingRound(_)))
                .count();
            rounds_done as f64 / self.round_count as f64
        };

        if actual_progress > 0.0 {
            target_progress / actual_progress
        } else {
            1.0
        }
    }

    /// Emit event
    async fn emit(&self, event: OrchestratorEvent) {
        let _ = self.event_tx.send(event).await;
    }

    /// Set current phase
    async fn set_phase(&self, phase: OrchestratedPhase, budget: Duration) {
        *self.current_phase.write().await = phase;
        self.emit(OrchestratorEvent::PhaseStarted { phase, budget }).await;
    }

    /// Record phase timing
    async fn record_phase(&self, phase: OrchestratedPhase, planned: Duration, actual: Duration) {
        let timing = PhaseTiming {
            phase,
            planned_duration: planned,
            actual_duration: actual,
            within_budget: actual <= planned,
            variance: actual.as_millis() as i64 - planned.as_millis() as i64,
        };

        self.phase_timings.write().await.push(timing.clone());
        self.emit(OrchestratorEvent::PhaseCompleted { phase, timing }).await;
    }

    /// Execute a phase with timing
    async fn execute_phase<F, Fut>(
        &self,
        phase: OrchestratedPhase,
        budget: Duration,
        action: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        if !self.should_continue().await {
            return Ok(());
        }

        self.set_phase(phase, budget).await;
        let start = Instant::now();

        // Execute the action
        let result = action().await;

        let actual = start.elapsed();

        // If we finished early, sleep to meet minimum visibility
        if actual < self.timing.min_phase_visibility {
            let remaining = self.timing.min_phase_visibility - actual;
            tokio::time::sleep(remaining).await;
        }

        // Record timing
        self.record_phase(phase, budget, start.elapsed()).await;

        result
    }

    /// Execute a timed delay phase
    async fn execute_delay_phase(&self, phase: OrchestratedPhase, duration: Duration) -> Result<()> {
        self.execute_phase(phase, duration, || async {
            tokio::time::sleep(duration).await;
            Ok(())
        })
        .await
    }

    /// Get snapshot of current state
    pub async fn snapshot(&self) -> OrchestratorSnapshot {
        let phase = *self.current_phase.read().await;
        let elapsed = self.elapsed().await;
        let remaining = self.remaining().await;
        let pace_factor = self.pace_factor().await;

        let client = self.rpc_client.read().await;
        OrchestratorSnapshot {
            phase,
            elapsed,
            remaining,
            on_schedule: pace_factor <= 1.2,
            pace_factor,
            completed_phases: self.phase_timings.read().await.clone(),
            training_progress: client.get_training_status().await.ok(),
            proof_progress: client.get_proof_status().await.ok(),
            workers: client.get_workers().await.unwrap_or_default(),
        }
    }

    /// Run the demo with timing guarantees
    pub async fn run(&self) -> Result<()> {
        // Mark start time
        *self.start_time.write().await = Some(Instant::now());

        // Initialize mock workers
        self.rpc_client.read().await.init_workers(self.worker_count).await;

        // === Setup Phase ===
        let setup_per_step = self.timing.setup_budget / 4;

        self.execute_delay_phase(OrchestratedPhase::Initializing, setup_per_step).await?;

        if !self.should_continue().await {
            return Ok(());
        }

        self.execute_delay_phase(OrchestratedPhase::DeployingContracts, setup_per_step).await?;
        self.execute_delay_phase(OrchestratedPhase::RegisteringModel, setup_per_step).await?;
        self.execute_delay_phase(OrchestratedPhase::StartingWorkers, setup_per_step / 2).await?;
        self.execute_delay_phase(OrchestratedPhase::Staking, setup_per_step / 2).await?;

        // === Training Phase ===
        // Initialize training tracker
        self.training_tracker.start(self.round_count as u64).await;

        // Start training (ignore errors in demo mode)
        let _ = self.rpc_client.read().await.start_training(self.round_count as u64).await;

        for round in 1..=self.round_count {
            if !self.should_continue().await {
                break;
            }

            // Adjust timing if we're behind schedule
            let pace = self.pace_factor().await;
            let round_budget = if self.timing.adaptive_pacing && pace > 1.2 {
                // We're behind, speed up
                Duration::from_secs_f64(self.timing.round_budget.as_secs_f64() / pace)
            } else {
                self.timing.round_budget
            };

            // Check if we need to trigger slashing
            if self.simulate_slashing && self.slashing_round == Some(round) {
                self.execute_slashing_event(round).await?;
            }

            // Training phase
            self.execute_phase(
                OrchestratedPhase::TrainingRound(round),
                round_budget * 4 / 10,
                || async {
                    self.rpc_client.read().await.set_phase(TrainingPhase::Forward).await;
                    tokio::time::sleep(round_budget / 10).await;

                    self.rpc_client.read().await.set_phase(TrainingPhase::Backward).await;
                    tokio::time::sleep(round_budget / 10).await;

                    self.rpc_client.read().await.set_phase(TrainingPhase::GradientCompute).await;
                    tokio::time::sleep(round_budget / 10).await;

                    self.rpc_client.read().await.advance_round().await;
                    Ok(())
                },
            )
            .await?;

            // Proof generation phase
            self.execute_phase(
                OrchestratedPhase::GeneratingProof(round),
                round_budget * 3 / 10,
                || async {
                    // Update proof tracker - witness generation
                    self.proof_tracker.update(ProofStatus {
                        generating: true,
                        phase: ProofPhase::WitnessGeneration,
                        progress_percent: 20,
                        constraints_satisfied: 20000,
                        total_constraints: 100000,
                        elapsed_ms: 0,
                        estimated_remaining_ms: 400,
                        memory_usage_bytes: 256 * 1024 * 1024,
                        gpu_accelerated: true,
                        error_bound: 0.0,
                    }).await;
                    self.rpc_client.read().await.set_proof_phase(ProofPhase::WitnessGeneration, 20).await;
                    tokio::time::sleep(round_budget / 15).await;

                    // Update proof tracker - commitment generation
                    self.proof_tracker.update(ProofStatus {
                        generating: true,
                        phase: ProofPhase::CommitmentGeneration,
                        progress_percent: 50,
                        constraints_satisfied: 50000,
                        total_constraints: 100000,
                        elapsed_ms: 150,
                        estimated_remaining_ms: 250,
                        memory_usage_bytes: 384 * 1024 * 1024,
                        gpu_accelerated: true,
                        error_bound: 0.0,
                    }).await;
                    self.rpc_client.read().await.set_proof_phase(ProofPhase::CommitmentGeneration, 50).await;
                    tokio::time::sleep(round_budget / 15).await;

                    // Update proof tracker - proof computation
                    self.proof_tracker.update(ProofStatus {
                        generating: true,
                        phase: ProofPhase::ProofComputation,
                        progress_percent: 80,
                        constraints_satisfied: 80000,
                        total_constraints: 100000,
                        elapsed_ms: 300,
                        estimated_remaining_ms: 100,
                        memory_usage_bytes: 512 * 1024 * 1024,
                        gpu_accelerated: true,
                        error_bound: 0.0,
                    }).await;
                    self.rpc_client.read().await.set_proof_phase(ProofPhase::ProofComputation, 80).await;
                    tokio::time::sleep(round_budget / 15).await;

                    // Update proof tracker - complete
                    self.proof_tracker.update(ProofStatus {
                        generating: false,
                        phase: ProofPhase::Complete,
                        progress_percent: 100,
                        constraints_satisfied: 100000,
                        total_constraints: 100000,
                        elapsed_ms: 400,
                        estimated_remaining_ms: 0,
                        memory_usage_bytes: 0,
                        gpu_accelerated: true,
                        error_bound: 0.0,
                    }).await;
                    self.rpc_client.read().await.set_proof_phase(ProofPhase::Complete, 100).await;

                    self.emit(OrchestratorEvent::ProofGenerated {
                        round,
                        time_ms: (round_budget.as_millis() * 3 / 10) as u64,
                    }).await;

                    Ok(())
                },
            )
            .await?;

            // Verification phase
            self.execute_phase(
                OrchestratedPhase::VerifyingProof(round),
                round_budget * 3 / 10,
                || async {
                    self.rpc_client.read().await.set_phase(TrainingPhase::ProofVerification).await;
                    tokio::time::sleep(round_budget / 5).await;

                    self.rpc_client.read().await.set_phase(TrainingPhase::WeightUpdate).await;

                    self.emit(OrchestratorEvent::ProofVerified {
                        round,
                        success: true,
                    }).await;

                    // Emit round completion
                    if let Ok(status) = self.rpc_client.read().await.get_training_status().await {
                        self.emit(OrchestratorEvent::RoundCompleted {
                            round,
                            loss: status.current_loss,
                            error_bound: status.accumulated_error,
                        }).await;

                        // Update training tracker
                        self.training_tracker.update_round(
                            round as u64,
                            status.current_loss,
                            status.accumulated_error,
                            TrainingPhase::RoundComplete,
                        ).await;
                    }

                    Ok(())
                },
            )
            .await?;

            // Check remaining time
            let remaining = self.remaining().await;
            if remaining < self.timing.finalization_budget {
                self.emit(OrchestratorEvent::TimingWarning {
                    message: "Low on time, may need to skip remaining rounds".to_string(),
                    remaining,
                }).await;

                // If critically low, skip to finalization
                if remaining < Duration::from_secs(5) {
                    break;
                }
            }
        }

        // === Finalization Phase ===
        if self.should_continue().await {
            self.execute_delay_phase(
                OrchestratedPhase::Finalizing,
                self.timing.finalization_budget,
            )
            .await?;
        }

        // === Complete ===
        let total_time = self.elapsed().await;
        *self.current_phase.write().await = OrchestratedPhase::Complete;

        self.emit(OrchestratorEvent::DemoCompleted {
            success: total_time <= self.timing.hard_deadline,
            total_time,
        }).await;

        Ok(())
    }

    /// Execute slashing event
    async fn execute_slashing_event(&self, round: u32) -> Result<()> {
        self.execute_phase(
            OrchestratedPhase::SlashingEvent,
            Duration::from_secs(3),
            || async {
                // Pick a worker to slash (not the first one)
                let worker_idx = if self.worker_count > 1 { 1 } else { 0 };

                self.rpc_client.read().await.slash_worker(worker_idx as usize).await;

                self.emit(OrchestratorEvent::WorkerSlashed {
                    worker_id: format!("worker-{}", worker_idx + 1),
                    reason: "Invalid proof submitted - error bound exceeded".to_string(),
                }).await;

                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(())
            },
        )
        .await
    }

    /// Get timing report
    pub async fn timing_report(&self) -> TimingReport {
        let timings = self.phase_timings.read().await.clone();
        let total_elapsed = self.elapsed().await;

        let total_variance: i64 = timings.iter().map(|t| t.variance).sum();
        let phases_over_budget = timings.iter().filter(|t| !t.within_budget).count();

        TimingReport {
            total_elapsed,
            target_duration: self.timing.total_duration,
            hard_deadline: self.timing.hard_deadline,
            within_target: total_elapsed <= self.timing.total_duration,
            within_deadline: total_elapsed <= self.timing.hard_deadline,
            total_variance_ms: total_variance,
            phases_over_budget,
            phase_timings: timings,
        }
    }
}

/// Timing report
#[derive(Debug, Clone)]
pub struct TimingReport {
    pub total_elapsed: Duration,
    pub target_duration: Duration,
    pub hard_deadline: Duration,
    pub within_target: bool,
    pub within_deadline: bool,
    pub total_variance_ms: i64,
    pub phases_over_budget: usize,
    pub phase_timings: Vec<PhaseTiming>,
}

impl TimingReport {
    /// Format as human-readable string
    pub fn display(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Demo Timing Report\n\
             ==================\n\
             Total Time: {:?} (target: {:?}, deadline: {:?})\n\
             Status: {}\n\
             Total Variance: {}ms\n\
             Phases Over Budget: {}\n\n\
             Phase Breakdown:\n",
            self.total_elapsed,
            self.target_duration,
            self.hard_deadline,
            if self.within_deadline { "SUCCESS" } else { "OVER TIME" },
            self.total_variance_ms,
            self.phases_over_budget,
        ));

        for timing in &self.phase_timings {
            output.push_str(&format!(
                "  {:30} planned: {:>6}ms, actual: {:>6}ms, variance: {:>+6}ms {}\n",
                timing.phase.name(),
                timing.planned_duration.as_millis(),
                timing.actual_duration.as_millis(),
                timing.variance,
                if timing.within_budget { "✓" } else { "⚠" },
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_config_default() {
        let config = DemoTimingConfig::default();
        assert!(config.total_duration <= MAX_DEMO_DURATION);
        assert!(config.hard_deadline <= Duration::from_secs(100));
    }

    #[test]
    fn test_timing_config_90_second() {
        let config = DemoTimingConfig::standard_90_second(10);

        // Should allocate ~6 seconds per round for training (70% of 85s / 10 rounds)
        let expected_round = Duration::from_secs_f64(85.0 * 0.7 / 10.0);
        let variance = (config.round_budget.as_secs_f64() - expected_round.as_secs_f64()).abs();
        assert!(variance < 1.0, "Round budget should be ~{:?}", expected_round);
    }

    #[test]
    fn test_phase_names() {
        assert_eq!(OrchestratedPhase::TrainingRound(1).name(), "Training Round 1");
        assert_eq!(OrchestratedPhase::GeneratingProof(5).name(), "Generating Proof 5");
    }

    #[test]
    fn test_phase_is_terminal() {
        assert!(!OrchestratedPhase::TrainingRound(1).is_terminal());
        assert!(OrchestratedPhase::Complete.is_terminal());
        assert!(OrchestratedPhase::Failed.is_terminal());
    }

    #[tokio::test]
    async fn test_mock_client_training() {
        let mock = MockRpcClient::new();
        mock.init_workers(3).await;
        mock.start_training(5).await;

        let status = mock.get_training_status().await;
        assert!(status.active);
        assert_eq!(status.total_rounds, 5);
    }
}
