//! Training Job Coordinator (aggregator side).
//!
//! Orchestrates a complete distributed training round:
//! 1. Configures the round (model, hyperparameters, dataset)
//! 2. Distributes model weights to workers
//! 3. Waits for workers to signal readiness
//! 4. Sends `BeginTraining` to kick off parallel training
//! 5. Collects `ProofSubmission` messages from workers
//! 6. Feeds proofs into `AggregatorNode` for aggregation
//! 7. Broadcasts `RoundCompleted`

use std::collections::HashMap;
use std::time::{Duration, Instant};

use helix_core::ModelCheckpoint;
use tokio::sync::mpsc;

use crate::network::messages::{
    MessagePayload, ModelDims, NetworkMessage, PeerId, RoundManagementMessage, TrainingMessage,
};
use crate::roles::aggregator::{AggregatorConfig, AggregatorNode, AggregatedResult};
use crate::trainer::MlpModel;

use super::CoordinationError;

/// Configuration for a training round.
#[derive(Debug, Clone)]
pub struct RoundConfig {
    /// Model to distribute to workers.
    pub model: MlpModel,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of training steps each worker should perform.
    pub steps_per_worker: u32,
    /// Maximum error budget for the round.
    pub error_budget: f64,
    /// Minimum number of workers required to proceed.
    pub min_workers: usize,
    /// Round deadline in seconds from now.
    pub deadline_secs: u64,
    /// On-chain model ID (for aggregation metadata).
    pub model_id: u64,
    /// Number of layers.
    pub num_layers: u32,
    /// Activation type (0=ReLU).
    pub activation_type: u8,
    /// Training dataset: list of (input, target) pairs.
    pub dataset: Vec<(Vec<f64>, Vec<f64>)>,
}

/// Result of a completed training round.
#[derive(Debug)]
pub struct RoundResult {
    /// The round ID.
    pub round_id: u64,
    /// Aggregated result from the aggregator.
    pub aggregated: AggregatedResult,
    /// Updated model after FedAvg aggregation.
    pub updated_model: Option<MlpModel>,
    /// Per-worker proof submission metadata.
    pub worker_submissions: Vec<WorkerSubmission>,
}

/// Metadata about a single worker's submission.
#[derive(Debug, Clone)]
pub struct WorkerSubmission {
    pub peer_id: PeerId,
    pub proof_len: usize,
    pub public_inputs_count: usize,
    pub error_bound: f64,
    pub steps_completed: u32,
    pub new_model_hash: [u8; 32],
}

/// Orchestrates distributed training from the aggregator's perspective.
pub struct TrainingJobCoordinator {
    /// Our peer ID.
    local_id: PeerId,
    /// Aggregator node for gradient collection + aggregation.
    aggregator: AggregatorNode,
    /// Send channels to each connected worker.
    worker_txs: HashMap<PeerId, mpsc::Sender<NetworkMessage>>,
    /// Receive channel for messages from all workers (tagged with sender).
    inbound_rx: mpsc::Receiver<(PeerId, NetworkMessage)>,
    /// Current round ID counter.
    round_counter: u64,
    /// Temporary storage for worker checkpoint data during proof collection.
    /// Used by `fedavg_from_submissions` after `collect_proofs` populates it.
    last_checkpoint_map: Option<HashMap<PeerId, Vec<u8>>>,
}

impl TrainingJobCoordinator {
    /// Creates a new coordinator.
    ///
    /// - `local_id`: this node's peer ID
    /// - `aggregator_config`: configuration for the underlying aggregator
    /// - `worker_txs`: pre-established send channels to workers
    /// - `inbound_rx`: multiplexed receive channel from all workers
    pub fn new(
        local_id: PeerId,
        aggregator_config: AggregatorConfig,
        worker_txs: HashMap<PeerId, mpsc::Sender<NetworkMessage>>,
        inbound_rx: mpsc::Receiver<(PeerId, NetworkMessage)>,
    ) -> Self {
        let aggregator = AggregatorNode::new(local_id.clone(), aggregator_config);
        Self {
            local_id,
            aggregator,
            worker_txs,
            inbound_rx,
            round_counter: 0,
            last_checkpoint_map: None,
        }
    }

    /// Returns a reference to the underlying aggregator.
    pub fn aggregator(&self) -> &AggregatorNode {
        &self.aggregator
    }

    /// Runs a complete training round.
    ///
    /// This is the main entry point: it distributes the model, collects proofs,
    /// runs aggregation, and returns the result.
    pub async fn run_training_round(
        &mut self,
        config: RoundConfig,
    ) -> Result<RoundResult, CoordinationError> {
        // Validate we have enough workers
        if self.worker_txs.len() < config.min_workers {
            return Err(CoordinationError::InsufficientWorkers {
                needed: config.min_workers,
                got: self.worker_txs.len(),
            });
        }

        self.round_counter += 1;
        let round_id = self.round_counter;
        let deadline = Instant::now() + Duration::from_secs(config.deadline_secs);

        tracing::info!(
            round_id,
            workers = self.worker_txs.len(),
            steps = config.steps_per_worker,
            "Starting distributed training round"
        );

        // Serialize dataset for workers (hex-encoded bincode)
        let dataset_bytes = bincode::serialize(&config.dataset)
            .map_err(|e| CoordinationError::CheckpointError(format!("dataset serialize: {}", e)))?;
        let dataset_ref = hex::encode(&dataset_bytes);

        // Serialize model checkpoint
        let checkpoint = config.model.to_checkpoint_with_arch(
            0,
            config.num_layers,
            config.activation_type,
        );
        let checkpoint_bytes = checkpoint.to_bytes()
            .map_err(|e| CoordinationError::CheckpointError(format!("checkpoint serialize: {}", e)))?;
        let model_hash = config.model.commitment();

        // Phase 1: Send RoundConfigure to all workers
        let round_configure = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::RoundManagement(RoundManagementMessage::RoundConfigure {
                round_id,
                model_id: config.model_id,
                model_dims: ModelDims {
                    d_in: config.model.d_in,
                    d_hid: config.model.d_hid,
                    d_out: config.model.d_out,
                    num_layers: config.num_layers,
                    num_heads: 0,
                    activation_type: config.activation_type,
                },
                dataset_ref,
                steps_per_worker: config.steps_per_worker,
                learning_rate: config.learning_rate,
                error_budget: config.error_budget,
                min_workers: config.min_workers as u32,
                deadline: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + config.deadline_secs,
                current_model_hash: model_hash,
                dataset_spec: None,
            }),
        );

        self.broadcast(&round_configure).await?;

        // Phase 2: Send ModelWeights to all workers
        let model_weights = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::Training(TrainingMessage::ModelWeights {
                round_id,
                checkpoint_data: checkpoint_bytes,
                weight_hash: model_hash,
            }),
        );

        self.broadcast(&model_weights).await?;

        // Phase 3: Wait for WorkerReady from enough workers
        let ready_workers = self
            .wait_for_ready(round_id, config.min_workers, deadline)
            .await?;

        tracing::info!(
            round_id,
            ready = ready_workers.len(),
            "Workers ready, starting training"
        );

        // Start a new aggregator round and register workers
        let training_params = crate::network::messages::TrainingParams {
            learning_rate: config.learning_rate,
            batch_size: 1,
            local_epochs: config.steps_per_worker,
            max_error_bound: config.error_budget,
            d_in: config.model.d_in,
            d_hid: config.model.d_hid,
            d_out: config.model.d_out,
            model_seed: 0,
            num_layers: config.num_layers,
            activation_type: config.activation_type,
        };
        self.aggregator.start_round(model_hash, training_params).await;

        for peer in &ready_workers {
            self.aggregator.register_worker(peer.clone()).await;
            self.aggregator
                .handle_participate_request(peer.clone(), round_id)
                .await;
        }
        self.aggregator.start_collection().await;

        // Phase 4: Send BeginTraining to ready workers
        let begin_training = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::RoundManagement(RoundManagementMessage::BeginTraining {
                round_id,
                participants: ready_workers.iter().map(|p| p.0.clone()).collect(),
            }),
        );

        for peer in &ready_workers {
            if let Some(tx) = self.worker_txs.get(peer) {
                let _ = tx.send(begin_training.clone()).await;
            }
        }

        // Phase 5: Collect ProofSubmissions
        let submissions = self
            .collect_proofs(round_id, ready_workers.len(), deadline)
            .await?;

        tracing::info!(
            round_id,
            submissions = submissions.len(),
            "All proofs collected, aggregating"
        );

        // Phase 6: Aggregate
        let aggregated = self
            .aggregator
            .aggregate()
            .await
            .ok_or_else(|| CoordinationError::AggregationFailed("aggregate returned None".into()))?;

        // Phase 7: FedAvg — average worker model checkpoints
        let updated_model = self.fedavg_from_submissions(&submissions);

        // Phase 8: Broadcast RoundCompleted
        let round_completed = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::RoundManagement(RoundManagementMessage::RoundCompleted {
                round_id,
                final_commitment: aggregated.commitment,
                tx_hash: None,
                new_model_hash: updated_model
                    .as_ref()
                    .map(|m| m.commitment())
                    .unwrap_or(aggregated.commitment),
                total_error_bound: aggregated.total_error_bound,
                num_contributors: aggregated.num_participants as u32,
            }),
        );

        self.broadcast(&round_completed).await?;

        tracing::info!(
            round_id,
            participants = aggregated.num_participants,
            error_bound = aggregated.total_error_bound,
            "Round completed"
        );

        Ok(RoundResult {
            round_id,
            aggregated,
            updated_model,
            worker_submissions: submissions,
        })
    }

    /// Broadcasts a message to all connected workers.
    async fn broadcast(&self, msg: &NetworkMessage) -> Result<(), CoordinationError> {
        for (peer_id, tx) in &self.worker_txs {
            if tx.send(msg.clone()).await.is_err() {
                tracing::warn!(%peer_id, "Failed to send to worker (channel closed)");
            }
        }
        Ok(())
    }

    /// Waits for `min_workers` to signal `WorkerReady`.
    async fn wait_for_ready(
        &mut self,
        round_id: u64,
        min_workers: usize,
        deadline: Instant,
    ) -> Result<Vec<PeerId>, CoordinationError> {
        let mut ready = Vec::new();

        while ready.len() < min_workers {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(CoordinationError::RoundTimeout {
                    elapsed_secs: 0,
                    deadline_secs: 0,
                });
            }

            match tokio::time::timeout(remaining, self.inbound_rx.recv()).await {
                Ok(Some((peer_id, msg))) => {
                    if let MessagePayload::RoundManagement(
                        RoundManagementMessage::WorkerReady {
                            round_id: rid,
                            ready: is_ready,
                            reason,
                        },
                    ) = &msg.payload
                    {
                        if *rid == round_id {
                            if *is_ready {
                                tracing::debug!(%peer_id, "Worker ready");
                                if !ready.contains(&peer_id) {
                                    ready.push(peer_id);
                                }
                            } else {
                                tracing::warn!(
                                    %peer_id,
                                    reason = reason.as_deref().unwrap_or("unknown"),
                                    "Worker not ready"
                                );
                            }
                        }
                    }
                }
                Ok(None) => {
                    return Err(CoordinationError::ChannelClosed(
                        "inbound channel closed".into(),
                    ));
                }
                Err(_) => {
                    return Err(CoordinationError::InsufficientWorkers {
                        needed: min_workers,
                        got: ready.len(),
                    });
                }
            }
        }

        Ok(ready)
    }

    /// Collects `ProofSubmission` messages from workers.
    async fn collect_proofs(
        &mut self,
        round_id: u64,
        expected: usize,
        deadline: Instant,
    ) -> Result<Vec<WorkerSubmission>, CoordinationError> {
        let mut submissions = Vec::new();
        let mut checkpoint_map: HashMap<PeerId, Vec<u8>> = HashMap::new();

        while submissions.len() < expected {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if !submissions.is_empty() {
                    tracing::warn!(
                        round_id,
                        got = submissions.len(),
                        expected,
                        "Deadline reached with partial submissions, proceeding"
                    );
                    break;
                }
                return Err(CoordinationError::RoundTimeout {
                    elapsed_secs: 0,
                    deadline_secs: 0,
                });
            }

            match tokio::time::timeout(remaining, self.inbound_rx.recv()).await {
                Ok(Some((peer_id, msg))) => {
                    if let MessagePayload::RoundManagement(
                        RoundManagementMessage::ProofSubmission {
                            round_id: rid,
                            proof,
                            public_inputs,
                            error_bound,
                            steps_completed,
                            new_model_hash,
                            checkpoint_data,
                        },
                    ) = msg.payload
                    {
                        if rid != round_id {
                            continue;
                        }

                        tracing::debug!(
                            %peer_id,
                            proof_len = proof.len(),
                            steps = steps_completed,
                            "Received proof submission"
                        );

                        // Convert public inputs to hex strings for the aggregator
                        let pi_hex: Vec<String> = public_inputs
                            .iter()
                            .map(|pi| format!("0x{}", hex::encode(pi)))
                            .collect();

                        // Feed into aggregator
                        let all_received = self
                            .aggregator
                            .handle_gradient_share_with_data(
                                peer_id.clone(),
                                round_id,
                                new_model_hash,
                                error_bound,
                                proof.clone(),
                                None,
                                Some(pi_hex),
                            )
                            .await;

                        // Store checkpoint for FedAvg
                        if !checkpoint_data.is_empty() {
                            checkpoint_map.insert(peer_id.clone(), checkpoint_data);
                        }

                        submissions.push(WorkerSubmission {
                            peer_id,
                            proof_len: proof.len(),
                            public_inputs_count: public_inputs.len(),
                            error_bound,
                            steps_completed,
                            new_model_hash,
                        });

                        if all_received {
                            tracing::info!(round_id, "All proofs received");
                            break;
                        }
                    }
                }
                Ok(None) => {
                    return Err(CoordinationError::ChannelClosed(
                        "inbound channel closed".into(),
                    ));
                }
                Err(_) => {
                    if !submissions.is_empty() {
                        tracing::warn!(
                            round_id,
                            got = submissions.len(),
                            expected,
                            "Timeout with partial submissions, proceeding"
                        );
                        break;
                    }
                    return Err(CoordinationError::RoundTimeout {
                        elapsed_secs: 0,
                        deadline_secs: 0,
                    });
                }
            }
        }

        self.last_checkpoint_map = Some(checkpoint_map);
        Ok(submissions)
    }

    /// Performs FedAvg: averages worker models from their checkpoint submissions.
    fn fedavg_from_submissions(
        &self,
        submissions: &[WorkerSubmission],
    ) -> Option<MlpModel> {
        let checkpoint_map = self.last_checkpoint_map.as_ref()?;
        if checkpoint_map.is_empty() {
            return None;
        }

        let mut models = Vec::new();
        for sub in submissions {
            if let Some(ckpt_bytes) = checkpoint_map.get(&sub.peer_id) {
                match ModelCheckpoint::from_bytes(ckpt_bytes) {
                    Ok(ckpt) => match MlpModel::from_checkpoint(&ckpt) {
                        Ok(model) => models.push(model),
                        Err(e) => {
                            tracing::warn!(
                                peer = %sub.peer_id,
                                "Failed to load model from checkpoint: {}", e
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            peer = %sub.peer_id,
                            "Failed to deserialize checkpoint: {}", e
                        );
                    }
                }
            }
        }

        if models.is_empty() {
            return None;
        }

        match crate::trainer::average_models(&models) {
            Ok(averaged) => Some(averaged),
            Err(e) => {
                tracing::error!("FedAvg failed: {}", e);
                None
            }
        }
    }
}
