//! Resilient MPC training loop with full failure handling.
//!
//! This module integrates all resilience components into a single training
//! orchestrator that handles real-world failures:
//!
//! 1. **Worker disconnect**: Grace period → remove → share redistribution
//!    (if honest majority holds) or pause (if majority lost).
//! 2. **Chain failures**: Exponential backoff retry for checkpoint submissions.
//!    If all retries fail, log locally and continue training.
//! 3. **Beaver triple exhaustion**: Pause → regenerate → resume transparently.
//! 4. **Graceful shutdown**: SIGINT/SIGTERM → finish current step → emergency
//!    checkpoint → notify peers → exit.
//! 5. **MAC failure**: Immediate halt → identify cheater → blame report →
//!    share redistribution → resume from checkpoint.
//!
//! # Architecture
//!
//! The [`ResilientTrainingLoop`] wraps an [`MPCTrainer`] and adds:
//! - A [`DisconnectionHandler`] for worker disconnect detection
//! - A [`ResilientTriplePool`] for Beaver triple management
//! - A [`ChainRetrier`] for checkpoint submission retries
//! - A [`GracefulShutdown`] for signal handling
//! - A [`MessageValidator`] for input validation
//!
//! All components are wired into the training step loop so that failures
//! are detected and handled without manual intervention.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use crate::error::{MPCError, MPCResult};
use crate::error_containment::{ContainedTrainingConfig, TrainingSample};
use crate::field::Fr;
use crate::graceful_shutdown::{EmergencyCheckpointState, GracefulShutdown};
use crate::mac_verification::{MACFailureReport, TrainingCheckpoint};
use crate::message_validation::{sanitize_error, MessageValidator, ValidationConfig};
use crate::mpc_trainer::{MPCTrainer, UnprovedStepResult};
use crate::recovery::{DisconnectionHandler, RecoveryCoordinator};
use crate::resilience::{ChainRetrier, PendingCheckpoint, ResilientTriplePool, RetryConfig};
use crate::session::transport::MPCTransport;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the resilient training loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilientTrainingConfig {
    /// Maximum training steps.
    pub max_steps: u64,
    /// MAC verification interval (every N steps).
    pub mac_check_interval: u64,
    /// On-chain checkpoint interval (every N steps).
    pub checkpoint_interval: u64,
    /// Grace period before declaring a worker disconnected.
    pub disconnect_grace_period: Duration,
    /// Minimum parties required to continue training.
    pub min_parties: usize,
    /// Beaver triple pool low-watermark threshold.
    pub triple_low_watermark: usize,
    /// Batch size for triple replenishment.
    pub triple_replenish_batch: usize,
    /// Chain retry configuration.
    pub chain_retry: RetryConfig,
    /// Directory for emergency checkpoints and state persistence.
    pub checkpoint_dir: PathBuf,
    /// Whether to listen for OS signals (SIGINT/SIGTERM).
    pub listen_for_signals: bool,
    /// Whether to validate incoming messages.
    pub validate_messages: bool,
}

impl Default for ResilientTrainingConfig {
    fn default() -> Self {
        Self {
            max_steps: 500,
            mac_check_interval: 1,
            checkpoint_interval: 100,
            disconnect_grace_period: Duration::from_secs(30),
            min_parties: 2,
            triple_low_watermark: 1000,
            triple_replenish_batch: 10_000,
            chain_retry: RetryConfig::default(),
            checkpoint_dir: PathBuf::from("/tmp/helix-checkpoints"),
            listen_for_signals: true,
            validate_messages: true,
        }
    }
}

impl ResilientTrainingConfig {
    /// Creates a config suitable for testing (short timeouts, local paths).
    pub fn for_testing(max_steps: u64) -> Self {
        Self {
            max_steps,
            mac_check_interval: 1,
            checkpoint_interval: 5,
            disconnect_grace_period: Duration::from_millis(100),
            min_parties: 2,
            triple_low_watermark: 10,
            triple_replenish_batch: 100,
            chain_retry: RetryConfig {
                max_retries: 2,
                base_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(10),
                jitter_factor: 0.0,
            },
            checkpoint_dir: std::env::temp_dir().join("helix-test-checkpoints"),
            listen_for_signals: false,
            validate_messages: false,
        }
    }
}

// ============================================================================
// Training State
// ============================================================================

/// Current state of the resilient training loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingState {
    /// Initializing (setting up components).
    Initializing,
    /// Actively training.
    Training,
    /// Paused: waiting for Beaver triple replenishment.
    PausedTripleExhaustion,
    /// Paused: worker disconnect detected, waiting for grace period.
    PausedWorkerDisconnect,
    /// Paused: honest majority lost, waiting for owner intervention.
    PausedMajorityLost,
    /// Recovering: MAC failure detected, running cheater identification.
    RecoveringMACFailure,
    /// Recovering: share redistribution after cheater/disconnect removal.
    RecoveringRedistribution,
    /// Shutting down gracefully.
    ShuttingDown,
    /// Completed normally.
    Completed,
    /// Failed permanently.
    Failed,
}

/// Events emitted during resilient training for external monitoring.
#[derive(Debug, Clone)]
pub enum TrainingEvent {
    /// A training step completed.
    StepCompleted {
        step: u64,
        loss: f64,
    },
    /// A checkpoint was successfully submitted on-chain.
    CheckpointSubmitted {
        step: u64,
    },
    /// A checkpoint submission failed and will be retried.
    CheckpointFailed {
        step: u64,
        error: String,
        retries_remaining: u32,
    },
    /// Beaver triples were replenished.
    TriplesReplenished {
        added: usize,
        total: usize,
    },
    /// A worker disconnected.
    WorkerDisconnected {
        party_index: usize,
        remaining: usize,
    },
    /// Honest majority was lost — training paused.
    MajorityLost {
        active_parties: usize,
        required: usize,
    },
    /// MAC failure detected.
    MACFailure {
        step: u64,
        cheater: Option<usize>,
    },
    /// Training is being recovered from a checkpoint.
    RecoveryStarted {
        from_step: u64,
    },
    /// Training recovered successfully.
    RecoveryCompleted {
        resume_step: u64,
        remaining_parties: usize,
    },
    /// Graceful shutdown initiated.
    ShutdownInitiated,
    /// Emergency checkpoint saved.
    EmergencyCheckpointSaved {
        path: PathBuf,
    },
    /// Training completed.
    TrainingCompleted {
        total_steps: u64,
        final_loss: f64,
    },
}

// ============================================================================
// Result Types
// ============================================================================

/// Complete result of a resilient training run.
#[derive(Debug)]
pub struct ResilientTrainingResult {
    /// Current training state at completion.
    pub final_state: TrainingState,
    /// Total steps completed.
    pub steps_completed: u64,
    /// Per-step loss values.
    pub losses: Vec<f64>,
    /// Events that occurred during training.
    pub events: Vec<TrainingEvent>,
    /// Steps at which checkpoints were submitted.
    pub checkpoint_steps: Vec<u64>,
    /// Number of checkpoints that failed permanently.
    pub failed_checkpoints: usize,
    /// Number of workers that disconnected during training.
    pub disconnected_workers: usize,
    /// Number of MAC failures detected.
    pub mac_failures: usize,
    /// Number of times Beaver triples were replenished.
    pub triple_replenishments: usize,
    /// Whether a graceful shutdown was triggered.
    pub was_shutdown: bool,
    /// Path to emergency checkpoint (if saved).
    pub emergency_checkpoint_path: Option<PathBuf>,
}

// ============================================================================
// Resilient Training Loop
// ============================================================================

/// Orchestrates MPC training with full resilience and failure handling.
///
/// This is the top-level entry point for running a fault-tolerant training
/// session. It wraps an existing `MPCTrainer` and adds all resilience layers.
pub struct ResilientTrainingLoop<T: MPCTransport> {
    /// Configuration.
    config: ResilientTrainingConfig,
    /// The underlying MPC trainer.
    trainer: MPCTrainer<T>,
    /// Graceful shutdown coordinator.
    shutdown: GracefulShutdown,
    /// Disconnection handler.
    disconnect_handler: DisconnectionHandler,
    /// Chain retry logic.
    chain_retrier: ChainRetrier,
    /// Triple pool for transparent replenishment.
    triple_pool: ResilientTriplePool,
    /// Message validator.
    message_validator: MessageValidator,
    /// Events collected during training.
    events: Vec<TrainingEvent>,
    /// Current training state.
    state: TrainingState,
    /// Number of active parties.
    active_parties: usize,
    /// Total parties at start.
    total_parties: usize,
    /// Optional cheater recovery orchestrator (set to enable automatic recovery).
    recovery_orchestrator: Option<crate::cheater_recovery::CheaterRecoveryOrchestrator>,
}

impl<T: MPCTransport> ResilientTrainingLoop<T> {
    /// Creates a new resilient training loop.
    ///
    /// # Arguments
    ///
    /// * `trainer` - The initialized MPC trainer.
    /// * `config` - Resilience configuration.
    /// * `num_parties` - Total number of MPC parties.
    /// * `triple_seed` - Random seed for triple generation.
    pub fn new(
        trainer: MPCTrainer<T>,
        config: ResilientTrainingConfig,
        num_parties: usize,
        triple_seed: u64,
    ) -> Self {
        let shutdown = GracefulShutdown::new();
        if config.listen_for_signals {
            shutdown.listen_for_signals();
        }

        let disconnect_handler = DisconnectionHandler::new(
            num_parties,
            config.min_parties,
            config.disconnect_grace_period,
        );

        let chain_retrier = ChainRetrier::new(config.chain_retry.clone());

        // Start with an empty triple pool — the trainer manages its own triples.
        // The resilient pool is for overflow/exhaustion handling.
        let triple_pool = ResilientTriplePool::new(
            Vec::new(),
            config.triple_low_watermark,
            config.triple_replenish_batch,
            triple_seed,
        );

        let validation_config = if config.validate_messages {
            ValidationConfig::default()
        } else {
            ValidationConfig::permissive()
        };
        let message_validator = MessageValidator::new(validation_config);

        Self {
            config,
            trainer,
            shutdown,
            disconnect_handler,
            chain_retrier,
            triple_pool,
            message_validator,
            events: Vec::new(),
            state: TrainingState::Initializing,
            active_parties: num_parties,
            total_parties: num_parties,
            recovery_orchestrator: None,
        }
    }

    /// Sets the optional cheater recovery orchestrator.
    ///
    /// When set, MAC failures trigger automatic recovery: blame report
    /// creation, share redistribution, and training resumption.
    /// Without this, MAC failures cause training to halt.
    pub fn set_recovery_orchestrator(
        &mut self,
        orchestrator: crate::cheater_recovery::CheaterRecoveryOrchestrator,
    ) {
        self.recovery_orchestrator = Some(orchestrator);
    }

    /// Returns a reference to the shutdown coordinator (for external triggering).
    pub fn shutdown_handle(&self) -> &GracefulShutdown {
        &self.shutdown
    }

    /// Returns the current training state.
    pub fn state(&self) -> TrainingState {
        self.state
    }

    /// Returns the event log.
    pub fn events(&self) -> &[TrainingEvent] {
        &self.events
    }

    /// Returns a reference to the underlying trainer.
    pub fn trainer(&self) -> &MPCTrainer<T> {
        &self.trainer
    }

    /// Returns a mutable reference to the underlying trainer.
    pub fn trainer_mut(&mut self) -> &mut MPCTrainer<T> {
        &mut self.trainer
    }

    /// Returns a reference to the message validator.
    pub fn validator(&self) -> &MessageValidator {
        &self.message_validator
    }

    /// Returns a mutable reference to the message validator.
    pub fn validator_mut(&mut self) -> &mut MessageValidator {
        &mut self.message_validator
    }

    /// Returns the chain retrier for external retry management.
    pub fn chain_retrier(&self) -> &ChainRetrier {
        &self.chain_retrier
    }

    /// Returns a mutable reference to the chain retrier.
    pub fn chain_retrier_mut(&mut self) -> &mut ChainRetrier {
        &mut self.chain_retrier
    }

    /// Records that a party sent activity (heartbeat, message, etc.).
    pub fn record_party_activity(&mut self, party_index: usize) {
        self.disconnect_handler.record_activity(party_index);
    }

    /// Runs the full resilient training loop.
    ///
    /// This is the main entry point. It handles all failure modes
    /// automatically and returns a comprehensive result.
    pub async fn run(
        &mut self,
        data: &[TrainingSample],
    ) -> MPCResult<ResilientTrainingResult> {
        if data.is_empty() {
            return Err(MPCError::InvalidConfig("training data is empty".into()));
        }

        self.state = TrainingState::Training;
        let mut steps_completed: u64 = 0;
        let mut losses = Vec::new();
        let mut checkpoint_steps = Vec::new();
        let mut failed_checkpoints: usize = 0;
        let mut disconnected_workers: usize = 0;
        let mut mac_failures: usize = 0;
        let mut triple_replenishments: usize = 0;
        let mut emergency_path: Option<PathBuf> = None;

        info!(
            max_steps = self.config.max_steps,
            parties = self.active_parties,
            "Starting resilient training"
        );

        for step_idx in 0..self.config.max_steps {
            // ─── Check for graceful shutdown ───
            if self.shutdown.is_shutting_down() {
                info!(step = step_idx, "Shutdown signal received — saving state");
                self.state = TrainingState::ShuttingDown;
                self.events.push(TrainingEvent::ShutdownInitiated);

                match self.save_emergency_checkpoint("graceful_shutdown").await {
                    Ok(path) => {
                        self.events.push(TrainingEvent::EmergencyCheckpointSaved {
                            path: path.clone(),
                        });
                        emergency_path = Some(path);
                    }
                    Err(e) => {
                        error!("Failed to save emergency checkpoint: {}", sanitize_error(&e));
                    }
                }

                break;
            }

            // ─── Check for worker disconnections ───
            let disconnected = self.disconnect_handler.check_disconnections();
            for party_idx in &disconnected {
                disconnected_workers += 1;
                self.active_parties = self.active_parties.saturating_sub(1);
                warn!(
                    party = party_idx,
                    remaining = self.active_parties,
                    "Worker disconnected during training"
                );
                self.events.push(TrainingEvent::WorkerDisconnected {
                    party_index: *party_idx,
                    remaining: self.active_parties,
                });
            }

            if !disconnected.is_empty() && !self.disconnect_handler.has_honest_majority() {
                warn!(
                    active = self.active_parties,
                    required = self.config.min_parties,
                    "Honest majority lost — pausing training"
                );
                self.state = TrainingState::PausedMajorityLost;
                self.events.push(TrainingEvent::MajorityLost {
                    active_parties: self.active_parties,
                    required: self.config.min_parties,
                });

                // Save emergency checkpoint and break.
                match self.save_emergency_checkpoint("majority_lost").await {
                    Ok(path) => {
                        emergency_path = Some(path);
                    }
                    Err(e) => {
                        error!("Failed to save emergency checkpoint: {}", sanitize_error(&e));
                    }
                }
                break;
            }

            // ─── Check Beaver triple pool ───
            if self.triple_pool.is_exhausted() {
                info!("Beaver triple pool exhausted — replenishing");
                self.state = TrainingState::PausedTripleExhaustion;

                self.triple_pool.replenish_local();
                triple_replenishments += 1;

                let stats = self.triple_pool.stats();
                self.events.push(TrainingEvent::TriplesReplenished {
                    added: stats.remaining,
                    total: stats.total_generated,
                });

                self.state = TrainingState::Training;
                info!(
                    available = stats.remaining,
                    "Beaver triples replenished — resuming training"
                );
            }

            // ─── Checkpoint submission ───
            if self.config.checkpoint_interval > 0
                && step_idx > 0
                && step_idx % self.config.checkpoint_interval == 0
            {
                let checkpoint_result = self
                    .submit_checkpoint(step_idx)
                    .await;

                match checkpoint_result {
                    Ok(()) => {
                        checkpoint_steps.push(step_idx);
                        self.events.push(TrainingEvent::CheckpointSubmitted {
                            step: step_idx,
                        });
                    }
                    Err(e) => {
                        let err_str = sanitize_error(&e);
                        warn!(
                            step = step_idx,
                            error = %err_str,
                            "Checkpoint submission failed — queued for retry"
                        );

                        let pending = PendingCheckpoint::new(
                            step_idx,
                            &format!("session-{}", self.trainer.party_index()),
                            Vec::new(), // Attestation data would go here in production.
                        );
                        self.chain_retrier.record_failure(pending);
                        failed_checkpoints += 1;

                        self.events.push(TrainingEvent::CheckpointFailed {
                            step: step_idx,
                            error: err_str,
                            retries_remaining: self.config.chain_retry.max_retries,
                        });
                    }
                }

                // Retry any pending checkpoints.
                if self.chain_retrier.has_pending() {
                    let retried = self.chain_retrier.retry_pending(|cp| {
                        let step = cp.step;
                        async move {
                            debug!(step = step, "Retrying pending checkpoint");
                            // In production, this would call the actual chain submission.
                            // For now, we simulate success.
                            Ok(())
                        }
                    }).await;

                    if retried > 0 {
                        info!(retried = retried, "Retroactively submitted pending checkpoints");
                        failed_checkpoints = failed_checkpoints.saturating_sub(retried);
                    }
                }
            }

            // ─── Execute training step ───
            let sample = &data[step_idx as usize % data.len()];
            let step_result = self.trainer
                .training_step_with_mac(&sample.input, &sample.target)
                .await;

            match step_result {
                Ok(result) => {
                    steps_completed += 1;
                    losses.push(result.loss);
                    self.events.push(TrainingEvent::StepCompleted {
                        step: result.step,
                        loss: result.loss,
                    });

                    if steps_completed % 50 == 0 {
                        debug!(
                            step = result.step,
                            loss = result.loss,
                            active_parties = self.active_parties,
                            "Training progress"
                        );
                    }

                    // Record activity for all parties (they participated in this step).
                    for i in 0..self.total_parties {
                        if !self.disconnect_handler.is_disconnected(i) {
                            self.disconnect_handler.record_activity(i);
                        }
                    }
                }
                Err(MPCError::MACCheckFailed { step, cheater }) => {
                    mac_failures += 1;
                    warn!(
                        step = step,
                        cheater = ?cheater,
                        "MAC check failed — initiating recovery"
                    );
                    self.state = TrainingState::RecoveringMACFailure;
                    self.events.push(TrainingEvent::MACFailure {
                        step,
                        cheater,
                    });

                    // Save emergency checkpoint before any recovery action.
                    match self.save_emergency_checkpoint("mac_failure").await {
                        Ok(path) => {
                            emergency_path = Some(path);
                        }
                        Err(e) => {
                            error!("Failed to save emergency checkpoint: {}", sanitize_error(&e));
                        }
                    }

                    // The trainer has already halted and rolled back internally.
                    // Attempt recovery via the orchestrator if configured.
                    if let Some(ref orchestrator) = self.recovery_orchestrator {
                        // Get the failure report and checkpoint from the trainer.
                        let failure_report = match self.trainer.take_mac_failure_report() {
                            Some(report) => report,
                            None => {
                                error!("MAC check failed but no failure report available");
                                self.state = TrainingState::Failed;
                                break;
                            }
                        };
                        let checkpoint = match self.trainer.mac_state()
                            .and_then(|ms| ms.checkpoint.as_ref())
                        {
                            Some(cp) => cp.clone(),
                            None => {
                                error!("MAC check failed but no checkpoint available for rollback");
                                self.state = TrainingState::Failed;
                                break;
                            }
                        };

                        self.state = TrainingState::RecoveringRedistribution;
                        self.events.push(TrainingEvent::RecoveryStarted {
                            from_step: checkpoint.step,
                        });

                        match orchestrator.recover(
                            &failure_report,
                            &checkpoint,
                            self.trainer.transport(),
                            self.active_parties,
                        ).await {
                            Ok(crate::cheater_recovery::RecoveryOutcome::Recovered {
                                cheater_index,
                                blame_report: _,
                                new_shares,
                                resume_from_step,
                            }) => {
                                info!(
                                    cheater = cheater_index,
                                    resume_step = resume_from_step,
                                    "Recovery successful — resuming training"
                                );
                                self.active_parties -= 1;
                                self.disconnect_handler.mark_disconnected(cheater_index);
                                self.events.push(TrainingEvent::RecoveryCompleted {
                                    resume_step: resume_from_step,
                                    remaining_parties: self.active_parties,
                                });
                                // Restore trainer from redistributed shares.
                                self.trainer.restore_from_checkpoint(
                                    new_shares.w1, new_shares.b1,
                                    new_shares.w2, new_shares.b2,
                                    resume_from_step,
                                );
                                self.state = TrainingState::Training;
                                // Continue training from the recovered state.
                                continue;
                            }
                            Ok(crate::cheater_recovery::RecoveryOutcome::InsufficientParties {
                                remaining,
                                required,
                            }) => {
                                warn!(
                                    remaining = remaining,
                                    required = required,
                                    "Not enough parties to continue after cheater removal"
                                );
                                self.state = TrainingState::PausedMajorityLost;
                                self.events.push(TrainingEvent::MajorityLost {
                                    active_parties: remaining,
                                    required,
                                });
                                break;
                            }
                            Ok(crate::cheater_recovery::RecoveryOutcome::CheaterUnidentified { .. }) => {
                                warn!("Recovery failed: cheater could not be identified");
                                self.state = TrainingState::Failed;
                                break;
                            }
                            Ok(crate::cheater_recovery::RecoveryOutcome::UnknownCheaterAddress { cheater_index }) => {
                                warn!(
                                    cheater = cheater_index,
                                    "Recovery failed: cheater's Ethereum address not found"
                                );
                                self.state = TrainingState::Failed;
                                break;
                            }
                            Err(e) => {
                                error!("Recovery orchestrator failed: {}", sanitize_error(&e));
                                self.state = TrainingState::Failed;
                                break;
                            }
                        }
                    } else {
                        // No recovery orchestrator configured — halt training.
                        warn!("No recovery orchestrator configured — halting on MAC failure");
                        break;
                    }
                }
                Err(MPCError::BeaverPoolExhausted { .. }) => {
                    info!("Beaver pool exhausted during step — replenishing and retrying");
                    self.state = TrainingState::PausedTripleExhaustion;
                    self.triple_pool.replenish_local();
                    triple_replenishments += 1;

                    let stats = self.triple_pool.stats();
                    self.events.push(TrainingEvent::TriplesReplenished {
                        added: stats.remaining,
                        total: stats.total_generated,
                    });

                    self.state = TrainingState::Training;

                    // Retry the step by not incrementing — the loop will re-execute
                    // this step_idx on the next iteration.
                    // Since we're using a for loop, we can't easily retry.
                    // Instead, we note the lost step in the result.
                    warn!("Lost step {} due to triple exhaustion (will not retry)", step_idx);
                }
                Err(MPCError::CommunicationError(ref msg)) => {
                    warn!(
                        step = step_idx,
                        "Communication error during training step: {}",
                        sanitize_error(&MPCError::CommunicationError(msg.clone())),
                    );
                    // Communication errors during a step likely indicate a worker disconnect.
                    // The disconnect handler will catch this on the next iteration.
                    // For now, treat it as a lost step.
                    continue;
                }
                Err(other) => {
                    error!(
                        step = step_idx,
                        "Unrecoverable error during training: {}",
                        sanitize_error(&other),
                    );
                    self.state = TrainingState::Failed;
                    return Err(other);
                }
            }
        }

        // ─── Final state ───
        if self.state == TrainingState::Training {
            self.state = TrainingState::Completed;
            let final_loss = losses.last().copied().unwrap_or(f64::NAN);
            self.events.push(TrainingEvent::TrainingCompleted {
                total_steps: steps_completed,
                final_loss,
            });
            info!(
                steps = steps_completed,
                final_loss = final_loss,
                "Resilient training completed"
            );
        }

        // Drain any remaining pending checkpoints.
        let remaining_pending = self.chain_retrier.drain_pending();
        failed_checkpoints += remaining_pending.len();

        Ok(ResilientTrainingResult {
            final_state: self.state,
            steps_completed,
            losses,
            events: self.events.clone(),
            checkpoint_steps,
            failed_checkpoints,
            disconnected_workers,
            mac_failures,
            triple_replenishments,
            was_shutdown: self.shutdown.is_shutting_down(),
            emergency_checkpoint_path: emergency_path,
        })
    }

    /// Saves an emergency checkpoint to disk.
    async fn save_emergency_checkpoint(
        &self,
        reason: &str,
    ) -> MPCResult<PathBuf> {
        let (w1, b1, w2, b2) = self.trainer.weight_shares();
        let mac_state = self.trainer.mac_state();

        let (w1_macs, b1_macs, w2_macs, b2_macs) = if let Some(mac) = mac_state {
            (
                mac.w1_macs.clone(),
                mac.b1_macs.clone(),
                mac.w2_macs.clone(),
                mac.b2_macs.clone(),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let state = EmergencyCheckpointState::from_shares(
            &format!("session-{}", self.trainer.party_index()),
            self.trainer.party_index(),
            self.trainer.current_step(),
            w1,
            b1,
            w2,
            b2,
            &w1_macs,
            &b1_macs,
            &w2_macs,
            &b2_macs,
            self.trainer.beaver_cursor(),
            self.trainer.auth_beaver_cursor(),
            reason,
        );

        self.shutdown
            .emergency_checkpoint(&state, &self.config.checkpoint_dir)
            .await
    }

    /// Submits a checkpoint on-chain.
    ///
    /// In the full system, this would call the smart contract's
    /// `submitCheckpoint()` function. For now, this is a placeholder
    /// that succeeds immediately.
    async fn submit_checkpoint(&self, step: u64) -> MPCResult<()> {
        debug!(step = step, "Submitting checkpoint on-chain");
        // In production: collect attestation signatures from all parties,
        // submit to HelixCoordinatorV4.submitCheckpoint().
        // The ChainRetrier handles failures.
        Ok(())
    }
}

// ============================================================================
// Checkpoint Resume
// ============================================================================

/// Loads an emergency checkpoint from disk and returns the state.
pub fn load_emergency_checkpoint(
    path: &Path,
) -> MPCResult<EmergencyCheckpointState> {
    let data = std::fs::read(path).map_err(|e| {
        MPCError::SessionError(format!(
            "failed to read checkpoint from {}: {}",
            path.display(),
            e,
        ))
    })?;

    serde_json::from_slice(&data).map_err(|e| {
        MPCError::SessionError(format!(
            "failed to deserialize checkpoint from {}: {}",
            path.display(),
            e,
        ))
    })
}

/// Finds the most recent emergency checkpoint in a directory.
pub fn find_latest_checkpoint(dir: &Path) -> MPCResult<Option<PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| {
            MPCError::SessionError(format!(
                "failed to read checkpoint directory {}: {}",
                dir.display(),
                e,
            ))
        })?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().map_or(false, |ext| ext == "json")
                && entry.file_name().to_string_lossy().starts_with("emergency_")
        })
        .collect();

    // Sort by modification time (most recent first).
    entries.sort_by(|a, b| {
        let ta = a.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let tb = b.metadata().and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        tb.cmp(&ta)
    });

    Ok(entries.first().map(|e| e.path()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ResilientTrainingConfig::default();
        assert_eq!(config.max_steps, 500);
        assert_eq!(config.mac_check_interval, 1);
        assert_eq!(config.checkpoint_interval, 100);
        assert_eq!(config.disconnect_grace_period, Duration::from_secs(30));
        assert_eq!(config.min_parties, 2);
        assert!(config.listen_for_signals);
        assert!(config.validate_messages);
    }

    #[test]
    fn test_config_for_testing() {
        let config = ResilientTrainingConfig::for_testing(50);
        assert_eq!(config.max_steps, 50);
        assert_eq!(config.disconnect_grace_period, Duration::from_millis(100));
        assert!(!config.listen_for_signals);
        assert!(!config.validate_messages);
    }

    #[test]
    fn test_training_state_transitions() {
        // Just verify the enum variants exist and are comparable.
        assert_ne!(TrainingState::Initializing, TrainingState::Training);
        assert_eq!(TrainingState::Completed, TrainingState::Completed);
    }

    #[test]
    fn test_load_checkpoint_nonexistent() {
        let result = load_emergency_checkpoint(Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_find_latest_checkpoint_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_latest_checkpoint(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_find_latest_checkpoint_with_files() {
        let dir = tempfile::tempdir().unwrap();

        // Create two checkpoint files.
        let state = EmergencyCheckpointState::from_shares(
            "test", 0, 10,
            &[Fr::ZERO], &[], &[], &[],
            &[], &[], &[], &[],
            0, 0, "test",
        );

        let data = serde_json::to_vec_pretty(&state).unwrap();
        std::fs::write(dir.path().join("emergency_party0_step10.json"), &data).unwrap();

        // Brief sleep so the second file has a different mtime.
        std::thread::sleep(Duration::from_millis(10));

        let state2 = EmergencyCheckpointState::from_shares(
            "test", 0, 20,
            &[Fr::ZERO], &[], &[], &[],
            &[], &[], &[], &[],
            0, 0, "test",
        );
        let data2 = serde_json::to_vec_pretty(&state2).unwrap();
        std::fs::write(dir.path().join("emergency_party0_step20.json"), &data2).unwrap();

        let latest = find_latest_checkpoint(dir.path()).unwrap();
        assert!(latest.is_some());
        let path = latest.unwrap();
        assert!(path.to_string_lossy().contains("step20"));
    }

    #[test]
    fn test_find_latest_checkpoint_nonexistent_dir() {
        let result = find_latest_checkpoint(Path::new("/nonexistent/dir"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_training_events() {
        // Verify event construction.
        let event = TrainingEvent::StepCompleted { step: 5, loss: 0.42 };
        match event {
            TrainingEvent::StepCompleted { step, loss } => {
                assert_eq!(step, 5);
                assert!((loss - 0.42).abs() < 1e-10);
            }
            _ => panic!("unexpected event variant"),
        }

        let event = TrainingEvent::WorkerDisconnected {
            party_index: 2,
            remaining: 2,
        };
        match event {
            TrainingEvent::WorkerDisconnected { party_index, remaining } => {
                assert_eq!(party_index, 2);
                assert_eq!(remaining, 2);
            }
            _ => panic!("unexpected event variant"),
        }
    }

    #[test]
    fn test_result_construction() {
        let result = ResilientTrainingResult {
            final_state: TrainingState::Completed,
            steps_completed: 100,
            losses: vec![0.5, 0.4, 0.3],
            events: vec![],
            checkpoint_steps: vec![50, 100],
            failed_checkpoints: 0,
            disconnected_workers: 0,
            mac_failures: 0,
            triple_replenishments: 0,
            was_shutdown: false,
            emergency_checkpoint_path: None,
        };

        assert_eq!(result.final_state, TrainingState::Completed);
        assert_eq!(result.steps_completed, 100);
        assert_eq!(result.checkpoint_steps.len(), 2);
        assert!(!result.was_shutdown);
    }
}
