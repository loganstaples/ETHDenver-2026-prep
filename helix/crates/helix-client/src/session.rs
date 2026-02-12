//! Training session with event streaming for the HELIX Client SDK.

use std::time::Duration;

use tokio::sync::mpsc;

use crate::error::HelixError;
use crate::model::TrainingParams;
use crate::rpc::client::{TrainingPhase, UnifiedRpcClient};

/// Events emitted during a training session.
#[derive(Debug, Clone)]
pub enum TrainingEvent {
    /// A new round has started.
    RoundStarted { round: u64, total: u64 },
    /// A training step within a round completed.
    StepCompleted { round: u64, loss: f64, error: f64 },
    /// A ZK proof was generated for the round.
    ProofGenerated { round: u64 },
    /// A proof was submitted on-chain.
    ProofSubmitted { round: u64, tx_hash: String },
    /// A round finished successfully.
    RoundCompleted { round: u64, loss: f64, error: f64 },
    /// All training rounds are done.
    TrainingComplete { final_loss: f64, rounds: u64 },
    /// An error occurred during training.
    Error { message: String },
}

/// Snapshot of training progress at a point in time.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    pub current_round: u64,
    pub total_rounds: u64,
    pub loss: f64,
    pub error: f64,
    pub phase: TrainingPhase,
    pub active: bool,
}

/// Final result of a completed training session.
#[derive(Debug, Clone)]
pub struct SessionResult {
    pub final_loss: f64,
    pub rounds_completed: u64,
    pub accumulated_error: f64,
    pub success: bool,
}

/// A streaming training session returned by [`HelixClient::start_training`].
///
/// Polls the node (or mock) for status updates and emits [`TrainingEvent`]s
/// that the caller can consume one at a time via [`next_event`](Self::next_event),
/// or block until completion via [`wait`](Self::wait).
pub struct TrainingSession {
    /// Receives events from the background polling task.
    event_rx: mpsc::Receiver<TrainingEvent>,
    /// Sends a cancellation signal to the polling task.
    cancel_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// Latest progress snapshot (updated by the polling task).
    progress: std::sync::Arc<tokio::sync::RwLock<TrainingProgress>>,
    /// Handle to the spawned polling task.
    _task: tokio::task::JoinHandle<()>,
}

impl TrainingSession {
    /// Create a new session and start the background polling task.
    pub(crate) fn new(
        rpc: UnifiedRpcClient,
        model_id: u64,
        params: TrainingParams,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel(64);
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();

        let progress = std::sync::Arc::new(tokio::sync::RwLock::new(TrainingProgress {
            current_round: 0,
            total_rounds: params.rounds,
            loss: 0.0,
            error: 0.0,
            phase: TrainingPhase::Initializing,
            active: true,
        }));

        let progress_clone = progress.clone();
        let poll_interval = Duration::from_millis(params.poll_interval_ms);

        let task = tokio::spawn(async move {
            Self::poll_loop(rpc, model_id, params, event_tx, cancel_rx, progress_clone, poll_interval).await;
        });

        Self {
            event_rx,
            cancel_tx: Some(cancel_tx),
            progress,
            _task: task,
        }
    }

    /// Background polling loop.
    async fn poll_loop(
        rpc: UnifiedRpcClient,
        model_id: u64,
        params: TrainingParams,
        event_tx: mpsc::Sender<TrainingEvent>,
        mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
        progress: std::sync::Arc<tokio::sync::RwLock<TrainingProgress>>,
        poll_interval: Duration,
    ) {
        let mut last_round: u64 = 0;
        let mut last_phase = TrainingPhase::Initializing;
        let mut consecutive_errors: u32 = 0;
        const MAX_CONSECUTIVE_ERRORS: u32 = 10;

        // Note: start_training_for() is called by HelixClient::start_training()
        // before creating this session. The poll_loop only needs to poll status.
        let _ = (model_id, &params); // suppress unused warnings

        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll_interval) => {}
                _ = &mut cancel_rx => {
                    let _ = event_tx
                        .send(TrainingEvent::Error {
                            message: "Training cancelled".to_string(),
                        })
                        .await;
                    return;
                }
            }

            // In mock mode, advance the simulation.
            if rpc.is_mock() {
                let continued = rpc.advance_round().await;
                if !continued {
                    // Training finished
                    let status = match rpc.get_training_status().await {
                        Ok(s) => s,
                        Err(_) => break,
                    };
                    let _ = event_tx
                        .send(TrainingEvent::TrainingComplete {
                            final_loss: status.current_loss,
                            rounds: params.rounds,
                        })
                        .await;

                    let mut p = progress.write().await;
                    p.active = false;
                    p.loss = status.current_loss;
                    p.error = status.accumulated_error;
                    p.current_round = status.current_round;
                    return;
                }
            }

            // Poll status from the node
            let status = match rpc.get_training_status().await {
                Ok(s) => {
                    consecutive_errors = 0;
                    s
                }
                Err(e) => {
                    consecutive_errors += 1;
                    let _ = event_tx
                        .send(TrainingEvent::Error {
                            message: e.to_string(),
                        })
                        .await;
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        let _ = event_tx
                            .send(TrainingEvent::Error {
                                message: format!(
                                    "Aborting after {} consecutive RPC errors",
                                    MAX_CONSECUTIVE_ERRORS
                                ),
                            })
                            .await;
                        return;
                    }
                    continue;
                }
            };

            // Update shared progress
            {
                let mut p = progress.write().await;
                p.current_round = status.current_round;
                p.total_rounds = status.total_rounds;
                p.loss = status.current_loss;
                p.error = status.accumulated_error;
                p.phase = status.phase;
                p.active = status.active;
            }

            // Detect round change
            if status.current_round > last_round {
                // Emit RoundStarted for the new round
                let _ = event_tx
                    .send(TrainingEvent::RoundStarted {
                        round: status.current_round,
                        total: status.total_rounds,
                    })
                    .await;

                // The previous round completed (if > 0)
                if last_round > 0 {
                    let _ = event_tx
                        .send(TrainingEvent::RoundCompleted {
                            round: last_round,
                            loss: status.current_loss,
                            error: status.accumulated_error,
                        })
                        .await;
                }

                // Emit step completed for the new round
                let _ = event_tx
                    .send(TrainingEvent::StepCompleted {
                        round: status.current_round,
                        loss: status.current_loss,
                        error: status.accumulated_error,
                    })
                    .await;

                last_round = status.current_round;
            }

            // Detect phase change
            if status.phase != last_phase {
                if status.phase == TrainingPhase::ProofGeneration {
                    let _ = event_tx
                        .send(TrainingEvent::ProofGenerated {
                            round: status.current_round,
                        })
                        .await;
                }
                if status.phase == TrainingPhase::ProofSubmission {
                    let _ = event_tx
                        .send(TrainingEvent::ProofSubmitted {
                            round: status.current_round,
                            tx_hash: format!("0xnode_{}", status.current_round),
                        })
                        .await;
                }
                last_phase = status.phase;
            }

            // Check if training is complete
            if !status.active
                || status.current_round >= status.total_rounds
            {
                // Emit final round completion
                if status.current_round > 0 && status.current_round > last_round.saturating_sub(1) {
                    let _ = event_tx
                        .send(TrainingEvent::RoundCompleted {
                            round: status.current_round,
                            loss: status.current_loss,
                            error: status.accumulated_error,
                        })
                        .await;
                }

                let _ = event_tx
                    .send(TrainingEvent::TrainingComplete {
                        final_loss: status.current_loss,
                        rounds: status.current_round,
                    })
                    .await;

                let mut p = progress.write().await;
                p.active = false;
                return;
            }
        }
    }

    /// Receive the next event, or `None` if the session is finished and the
    /// channel is closed.
    pub async fn next_event(&mut self) -> Option<TrainingEvent> {
        self.event_rx.recv().await
    }

    /// Block until training completes and return the final result.
    pub async fn wait(&mut self) -> Result<SessionResult, HelixError> {
        let mut last_loss = 0.0;
        let mut last_rounds = 0u64;
        let mut last_error = 0.0;

        while let Some(event) = self.next_event().await {
            match event {
                TrainingEvent::TrainingComplete { final_loss, rounds } => {
                    let p = self.progress.read().await;
                    return Ok(SessionResult {
                        final_loss,
                        rounds_completed: rounds,
                        accumulated_error: p.error,
                        success: true,
                    });
                }
                TrainingEvent::Error { message } if message == "Training cancelled" => {
                    return Ok(SessionResult {
                        final_loss: last_loss,
                        rounds_completed: last_rounds,
                        accumulated_error: last_error,
                        success: false,
                    });
                }
                TrainingEvent::StepCompleted { round, loss, error } => {
                    last_loss = loss;
                    last_rounds = round;
                    last_error = error;
                }
                TrainingEvent::RoundCompleted { loss, error, round } => {
                    last_loss = loss;
                    last_rounds = round;
                    last_error = error;
                }
                _ => {}
            }
        }

        // Channel closed without TrainingComplete — unusual
        Ok(SessionResult {
            final_loss: last_loss,
            rounds_completed: last_rounds,
            accumulated_error: last_error,
            success: false,
        })
    }

    /// Cancel the training session.
    pub fn cancel(&mut self) {
        if let Some(tx) = self.cancel_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Get a snapshot of the current progress.
    pub async fn progress(&self) -> TrainingProgress {
        self.progress.read().await.clone()
    }
}
