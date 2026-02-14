//! Training Worker Handler (worker/compute side).
//!
//! Processes coordinator messages and runs real ML training with ZK proofs:
//! 1. Receives `RoundConfigure` with model dimensions + hyperparameters
//! 2. Receives `ModelWeights` with serialized checkpoint
//! 3. Loads model, signals `WorkerReady`
//! 4. On `BeginTraining`, runs `Trainer::train()` with real ZK proof generation
//! 5. Sends `ProofSubmission` with proof bytes, public inputs, and updated checkpoint

use helix_core::ModelCheckpoint;
use tokio::sync::mpsc;

use crate::network::messages::{
    MessagePayload, NetworkMessage, PeerId, RoundManagementMessage, TrainingMessage,
};
use crate::trainer::Trainer;

use super::CoordinationError;

/// State accumulated during round configuration before training begins.
struct RoundState {
    round_id: u64,
    learning_rate: f64,
    steps_per_worker: u32,
    #[allow(dead_code)]
    error_budget: f64,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    num_layers: u32,
    activation_type: u8,
    dataset_ref: String,
    model_checkpoint: Option<ModelCheckpoint>,
}

/// Handles coordinator messages and runs training on the worker side.
pub struct TrainingWorkerHandler {
    /// Our peer ID.
    local_id: PeerId,
    /// Channel to send messages to the coordinator.
    outbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
    /// Channel to receive messages from the coordinator.
    inbound_rx: mpsc::Receiver<NetworkMessage>,
}

impl TrainingWorkerHandler {
    /// Creates a new worker handler.
    ///
    /// - `local_id`: this worker's peer ID
    /// - `outbound_tx`: send channel to coordinator (messages tagged with our peer ID)
    /// - `inbound_rx`: receive channel from coordinator
    pub fn new(
        local_id: PeerId,
        outbound_tx: mpsc::Sender<(PeerId, NetworkMessage)>,
        inbound_rx: mpsc::Receiver<NetworkMessage>,
    ) -> Self {
        Self {
            local_id,
            outbound_tx,
            inbound_rx,
        }
    }

    /// Runs one complete training round.
    ///
    /// Blocks until the round completes or an error occurs.
    /// Returns the number of training steps completed and proof bytes produced.
    pub async fn run_round(&mut self) -> Result<WorkerRoundResult, CoordinationError> {
        // Phase 1: Wait for RoundConfigure
        let mut round_state = self.wait_for_round_configure().await?;

        tracing::info!(
            round_id = round_state.round_id,
            worker = %self.local_id,
            "Received round configuration"
        );

        // Phase 2: Wait for ModelWeights
        self.wait_for_model_weights(&mut round_state).await?;

        // Load the model from checkpoint
        let checkpoint = round_state.model_checkpoint.as_ref()
            .ok_or_else(|| CoordinationError::CheckpointError("no checkpoint received".into()))?;

        let trainer = Trainer::from_checkpoint_with_params(
            checkpoint,
            round_state.learning_rate,
            round_state.d_in,
            round_state.d_hid,
            round_state.d_out,
            round_state.num_layers,
            round_state.activation_type,
        ).map_err(|e| CoordinationError::CheckpointError(format!("load model: {}", e)))?;

        // Signal ready
        let ready_msg = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::RoundManagement(RoundManagementMessage::WorkerReady {
                round_id: round_state.round_id,
                ready: true,
                reason: None,
            }),
        );
        self.send(ready_msg).await?;

        tracing::info!(
            round_id = round_state.round_id,
            worker = %self.local_id,
            "Signaled ready, waiting for BeginTraining"
        );

        // Phase 3: Wait for BeginTraining
        self.wait_for_begin_training(round_state.round_id).await?;

        tracing::info!(
            round_id = round_state.round_id,
            worker = %self.local_id,
            steps = round_state.steps_per_worker,
            "Training started"
        );

        // Phase 4: Decode dataset and run training with proof generation
        let dataset = self.decode_dataset(&round_state.dataset_ref)?;

        // Run training on a blocking thread (proof generation is CPU-intensive)
        let round_id = round_state.round_id;
        let num_steps = round_state.steps_per_worker as usize;
        let num_layers = round_state.num_layers;
        let activation_type = round_state.activation_type;

        let training_result = tokio::task::spawn_blocking(move || {
            run_training(trainer, &dataset, num_steps, num_layers, activation_type)
        })
        .await
        .map_err(|e| CoordinationError::ProofFailed(format!("spawn_blocking join: {}", e)))?
        .map_err(|e| CoordinationError::ProofFailed(e.to_string()))?;

        tracing::info!(
            round_id,
            worker = %self.local_id,
            steps = training_result.steps_completed,
            proof_len = training_result.proof.len(),
            loss = training_result.loss,
            "Training complete, submitting proof"
        );

        // Phase 5: Send ProofSubmission
        let submission = NetworkMessage::new(
            self.local_id.clone(),
            MessagePayload::RoundManagement(RoundManagementMessage::ProofSubmission {
                round_id,
                proof: training_result.proof.clone(),
                public_inputs: training_result.public_inputs.clone(),
                error_bound: training_result.error_bound,
                steps_completed: training_result.steps_completed,
                new_model_hash: training_result.new_model_hash,
                checkpoint_data: training_result.checkpoint_data.clone(),
            }),
        );
        self.send(submission).await?;

        // Phase 6: Wait for RoundCompleted (optional, don't fail if timeout)
        let final_commitment = self.wait_for_round_completed(round_id).await.ok();

        Ok(WorkerRoundResult {
            round_id,
            steps_completed: training_result.steps_completed,
            proof_len: training_result.proof.len(),
            public_inputs_count: training_result.public_inputs.len(),
            loss: training_result.loss,
            error_bound: training_result.error_bound,
            new_model_hash: training_result.new_model_hash,
            final_commitment,
        })
    }

    /// Sends a message to the coordinator.
    async fn send(&self, msg: NetworkMessage) -> Result<(), CoordinationError> {
        self.outbound_tx
            .send((self.local_id.clone(), msg))
            .await
            .map_err(|_| CoordinationError::ChannelClosed("outbound channel closed".into()))
    }

    /// Waits for `RoundConfigure` message.
    async fn wait_for_round_configure(&mut self) -> Result<RoundState, CoordinationError> {
        loop {
            let msg = self.inbound_rx.recv().await
                .ok_or_else(|| CoordinationError::ChannelClosed("inbound closed waiting for RoundConfigure".into()))?;

            if let MessagePayload::RoundManagement(
                RoundManagementMessage::RoundConfigure {
                    round_id,
                    model_dims,
                    dataset_ref,
                    steps_per_worker,
                    learning_rate,
                    error_budget,
                    ..
                },
            ) = msg.payload
            {
                return Ok(RoundState {
                    round_id,
                    learning_rate,
                    steps_per_worker,
                    error_budget,
                    d_in: model_dims.d_in,
                    d_hid: model_dims.d_hid,
                    d_out: model_dims.d_out,
                    num_layers: model_dims.num_layers,
                    activation_type: model_dims.activation_type,
                    dataset_ref,
                    model_checkpoint: None,
                });
            }
        }
    }

    /// Waits for `ModelWeights` message and loads the checkpoint.
    async fn wait_for_model_weights(
        &mut self,
        state: &mut RoundState,
    ) -> Result<(), CoordinationError> {
        loop {
            let msg = self.inbound_rx.recv().await
                .ok_or_else(|| CoordinationError::ChannelClosed("inbound closed waiting for ModelWeights".into()))?;

            if let MessagePayload::Training(TrainingMessage::ModelWeights {
                round_id,
                checkpoint_data,
                weight_hash: _,
            }) = msg.payload
            {
                if round_id == state.round_id {
                    let checkpoint = ModelCheckpoint::from_bytes(&checkpoint_data)
                        .map_err(|e| CoordinationError::CheckpointError(
                            format!("deserialize checkpoint: {}", e),
                        ))?;
                    state.model_checkpoint = Some(checkpoint);
                    return Ok(());
                }
            }
        }
    }

    /// Waits for `BeginTraining` message.
    async fn wait_for_begin_training(
        &mut self,
        round_id: u64,
    ) -> Result<(), CoordinationError> {
        loop {
            let msg = self.inbound_rx.recv().await
                .ok_or_else(|| CoordinationError::ChannelClosed("inbound closed waiting for BeginTraining".into()))?;

            if let MessagePayload::RoundManagement(
                RoundManagementMessage::BeginTraining {
                    round_id: rid,
                    ..
                },
            ) = &msg.payload
            {
                if *rid == round_id {
                    return Ok(());
                }
            }
        }
    }

    /// Waits for `RoundCompleted` message (non-blocking timeout).
    async fn wait_for_round_completed(
        &mut self,
        round_id: u64,
    ) -> Result<[u8; 32], CoordinationError> {
        let timeout = tokio::time::Duration::from_secs(30);
        loop {
            match tokio::time::timeout(timeout, self.inbound_rx.recv()).await {
                Ok(Some(msg)) => {
                    if let MessagePayload::RoundManagement(
                        RoundManagementMessage::RoundCompleted {
                            round_id: rid,
                            final_commitment,
                            ..
                        },
                    ) = &msg.payload
                    {
                        if *rid == round_id {
                            return Ok(*final_commitment);
                        }
                    }
                }
                Ok(None) => {
                    return Err(CoordinationError::ChannelClosed("channel closed".into()));
                }
                Err(_) => {
                    return Err(CoordinationError::RoundTimeout {
                        elapsed_secs: 30,
                        deadline_secs: 30,
                    });
                }
            }
        }
    }

    /// Decodes the hex-encoded bincode dataset from the coordinator.
    fn decode_dataset(
        &self,
        dataset_ref: &str,
    ) -> Result<Vec<(Vec<f64>, Vec<f64>)>, CoordinationError> {
        let bytes = hex::decode(dataset_ref)
            .map_err(|e| CoordinationError::CheckpointError(format!("hex decode dataset: {}", e)))?;
        let dataset: Vec<(Vec<f64>, Vec<f64>)> = bincode::deserialize(&bytes)
            .map_err(|e| CoordinationError::CheckpointError(format!("bincode decode dataset: {}", e)))?;
        Ok(dataset)
    }
}

/// Result of training on the worker side, ready to send as ProofSubmission.
struct TrainingOutput {
    steps_completed: u32,
    proof: Vec<u8>,
    public_inputs: Vec<[u8; 32]>,
    error_bound: f64,
    loss: f64,
    new_model_hash: [u8; 32],
    checkpoint_data: Vec<u8>,
}

/// Result returned to the caller after a worker completes a round.
#[derive(Debug)]
pub struct WorkerRoundResult {
    pub round_id: u64,
    pub steps_completed: u32,
    pub proof_len: usize,
    pub public_inputs_count: usize,
    pub loss: f64,
    pub error_bound: f64,
    pub new_model_hash: [u8; 32],
    pub final_commitment: Option<[u8; 32]>,
}

/// Runs training and proof generation (called on a blocking thread).
///
/// Executes `num_steps` training steps with real ZK proof generation via
/// `MLTrainingProverV2`. Returns the last step's proof + public inputs.
fn run_training(
    mut trainer: Trainer,
    dataset: &[(Vec<f64>, Vec<f64>)],
    num_steps: usize,
    num_layers: u32,
    activation_type: u8,
) -> anyhow::Result<TrainingOutput> {
    let (proved_steps, _metrics) = trainer.train(dataset, num_steps)?;

    // Use the last proved step's proof and public inputs
    let last_step = proved_steps.last()
        .ok_or_else(|| anyhow::anyhow!("no proved steps produced"))?;

    // Extract public inputs as [u8; 32] from the proof result
    let public_inputs: Vec<[u8; 32]> = last_step.proof_result.public_inputs
        .iter()
        .map(|fr| {
            use helix_prover::halo2curves::ff::PrimeField;
            let repr = fr.to_repr();
            let mut bytes = [0u8; 32];
            bytes.copy_from_slice(repr.as_ref());
            bytes
        })
        .collect();

    // Compute error bound from accumulated quantization error
    let error_bound = trainer.accumulated_error();

    // Serialize updated model checkpoint
    let model = trainer.model();
    let checkpoint = model.to_checkpoint_with_arch(
        trainer.step_count(),
        num_layers,
        activation_type,
    );
    let checkpoint_data = checkpoint.to_bytes()
        .map_err(|e| anyhow::anyhow!("checkpoint serialize: {}", e))?;

    Ok(TrainingOutput {
        steps_completed: num_steps as u32,
        proof: last_step.proof_result.proof.clone(),
        public_inputs,
        error_bound,
        loss: last_step.loss,
        new_model_hash: model.commitment(),
        checkpoint_data,
    })
}
