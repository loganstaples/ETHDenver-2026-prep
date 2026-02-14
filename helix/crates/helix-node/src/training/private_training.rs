//! Private Training Integration.
//!
//! Integrates helix-mpc's `MPCTrainer` into the node's training pipeline via
//! `NodeTransport`. Enables privacy-preserving distributed training where:
//!
//! - Model weights are secret-shared across workers
//! - Forward/backward passes use Beaver-triple multiplication (weights stay shared)
//! - ReLU activations use secure sign-bit evaluation (no cleartext reconstruction)
//! - Gradients remain secret-shared; weight updates are local per-share operations
//! - Periodic re-sharing prevents gradient accumulation attacks
//!
//! # Usage
//!
//! ```ignore
//! let config = PrivateTrainingConfig {
//!     mode: TrainingMode::Private,
//!     d_in: 2, d_hid: 4, d_out: 1,
//!     num_parties: 3,
//!     ..Default::default()
//! };
//! let model = MlpModel::new_random(2, 4, 1, 42);
//! let results = run_private_training_round(
//!     config, &model, &samples, "session-1",
//! ).await?;
//! ```

use tokio::task::JoinHandle;
use tracing::{debug, error, info, instrument, warn};

use helix_mpc::mpc_trainer::{
    MPCTrainer, MPCTrainerConfig, MPCTrainingStepResult,
    ModelWeights as MpcModelWeights,
};
use helix_mpc::session::node_transport::{NodeTransport, MPCMessageRouter};
use helix_mpc::error::{MPCError, MPCResult};
use helix_mpc::types::PartyId;
use helix_mpc::Fr;

use crate::trainer::MlpModel;

// ──────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────

/// Whether training uses MPC privacy or runs in cleartext.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMode {
    /// Full MPC: secret-shared weights, garbled-circuit ReLU, Beaver-triple
    /// multiplication. Provides cryptographic privacy guarantees.
    Private,
    /// Cleartext: standard training with no privacy. Used for testing,
    /// benchmarking, or when privacy is not required.
    Cleartext,
}

impl Default for TrainingMode {
    fn default() -> Self {
        Self::Private
    }
}

/// Configuration for private (MPC-backed) training.
#[derive(Debug, Clone)]
pub struct PrivateTrainingConfig {
    /// Training mode (Private or Cleartext).
    pub mode: TrainingMode,
    /// Input dimension.
    pub d_in: usize,
    /// Hidden dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of MPC parties (workers).
    pub num_parties: usize,
    /// Re-share weights every N steps (0 = disabled).
    pub reshare_interval: u64,
    /// Number of Beaver triples to pre-generate per batch.
    pub beaver_batch_size: usize,
    /// Whether to generate ZK proofs after each training step.
    pub generate_proofs: bool,
    /// Base error bound per operation.
    pub base_error: f64,
}

impl Default for PrivateTrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::Private,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            learning_rate: 0.01,
            num_parties: 3,
            reshare_interval: 50,
            beaver_batch_size: 256,
            generate_proofs: false,
            base_error: 1e-6,
        }
    }
}

impl PrivateTrainingConfig {
    /// Creates a small config suitable for testing.
    pub fn small(num_parties: usize) -> Self {
        Self {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            num_parties,
            ..Default::default()
        }
    }

    /// Converts to the MPC trainer config used by helix-mpc.
    pub fn to_mpc_config(&self) -> MPCTrainerConfig {
        MPCTrainerConfig {
            d_in: self.d_in,
            d_hid: self.d_hid,
            d_out: self.d_out,
            learning_rate: self.learning_rate,
            num_parties: self.num_parties,
            reshare_interval: self.reshare_interval,
            beaver_batch_size: self.beaver_batch_size,
            generate_proofs: self.generate_proofs,
            base_error: self.base_error,
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Model conversion
// ──────────────────────────────────────────────────────────────

/// Converts a node `MlpModel` (f64) to MPC `ModelWeights` (Fr field elements).
pub fn mlp_to_mpc_weights(model: &MlpModel) -> MpcModelWeights {
    MpcModelWeights {
        w1: model.w1.iter().map(|&v| Fr::from_f64(v)).collect(),
        b1: model.b1.iter().map(|&v| Fr::from_f64(v)).collect(),
        w2: model.w2.iter().map(|&v| Fr::from_f64(v)).collect(),
        b2: model.b2.iter().map(|&v| Fr::from_f64(v)).collect(),
    }
}

/// Converts MPC `ModelWeights` (Fr field elements) back to a node `MlpModel` (f64).
pub fn mpc_to_mlp_model(
    weights: &MpcModelWeights,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> MlpModel {
    MlpModel::new(
        d_in,
        d_hid,
        d_out,
        weights.w1.iter().map(|v| v.to_f64()).collect(),
        weights.b1.iter().map(|v| v.to_f64()).collect(),
        weights.w2.iter().map(|v| v.to_f64()).collect(),
        weights.b2.iter().map(|v| v.to_f64()).collect(),
    )
}

// ──────────────────────────────────────────────────────────────
// Private Training Session (aggregator side)
// ──────────────────────────────────────────────────────────────

/// A private training session backed by MPC.
///
/// Created by the aggregator when starting a training round with MPC enabled.
/// Manages the transport mesh and distributes transports to workers.
pub struct PrivateTrainingSession {
    /// Session identifier.
    session_id: String,
    /// Configuration.
    config: PrivateTrainingConfig,
    /// The initial model weights in MPC format.
    initial_weights: MpcModelWeights,
    /// Pre-created transports (one per party). Taken by workers via `take_worker_transport`.
    transports: Vec<Option<NodeTransport>>,
    /// MPC message router for incoming message dispatch.
    router: MPCMessageRouter,
}

impl PrivateTrainingSession {
    /// Creates a new private training session.
    ///
    /// This creates a mesh of `NodeTransport` instances (one per party) connected
    /// by in-memory channels. In a real multi-node deployment, these channels
    /// would be bridged to the P2P network via `ConnectionPoolBridge`.
    pub fn new(
        config: PrivateTrainingConfig,
        model: &MlpModel,
        session_id: impl Into<String>,
    ) -> MPCResult<Self> {
        let session_id = session_id.into();
        let n = config.num_parties;

        if n < 2 {
            return Err(MPCError::InvalidConfig(
                "MPC requires at least 2 parties".into(),
            ));
        }

        let initial_weights = mlp_to_mpc_weights(model);

        // Create party IDs.
        let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();

        // Create a fully connected mesh of NodeTransports.
        let transport_vec = NodeTransport::create_mesh(&parties, &session_id);

        let transports: Vec<Option<NodeTransport>> = transport_vec
            .into_iter()
            .map(Some)
            .collect();

        let router = MPCMessageRouter::new();

        info!(
            session_id = %session_id,
            num_parties = n,
            d_in = config.d_in,
            d_hid = config.d_hid,
            d_out = config.d_out,
            "Private training session created"
        );

        Ok(Self {
            session_id,
            config,
            initial_weights,
            transports,
            router,
        })
    }

    /// Takes the transport for a specific party index.
    ///
    /// Each transport can only be taken once. Returns `None` if already taken
    /// or if the index is out of range.
    pub fn take_worker_transport(&mut self, party_index: usize) -> Option<NodeTransport> {
        self.transports
            .get_mut(party_index)
            .and_then(|slot| slot.take())
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the number of parties.
    pub fn num_parties(&self) -> usize {
        self.config.num_parties
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &PrivateTrainingConfig {
        &self.config
    }

    /// Returns a reference to the initial MPC weights.
    pub fn initial_weights(&self) -> &MpcModelWeights {
        &self.initial_weights
    }

    /// Returns a reference to the message router.
    pub fn router(&self) -> &MPCMessageRouter {
        &self.router
    }

    /// Returns the number of transports that haven't been taken yet.
    pub fn available_transports(&self) -> usize {
        self.transports.iter().filter(|t| t.is_some()).count()
    }
}

// ──────────────────────────────────────────────────────────────
// Private Training Worker (per-worker handle)
// ──────────────────────────────────────────────────────────────

/// A worker in a private training session.
///
/// Wraps `MPCTrainer<NodeTransport>` and provides a high-level interface
/// for the node's training pipeline. Each worker holds one of these.
pub struct PrivateTrainingWorker {
    /// The inner MPC trainer.
    trainer: MPCTrainer<NodeTransport>,
    /// Configuration.
    config: PrivateTrainingConfig,
    /// This worker's party index.
    party_index: usize,
    /// Whether weight sharing has been done.
    weights_shared: bool,
    /// Whether Beaver triples have been generated.
    triples_generated: bool,
}

impl PrivateTrainingWorker {
    /// Creates a new private training worker.
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration
    /// * `transport` - Pre-wired `NodeTransport` from the session
    /// * `party_index` - This worker's party index (0..num_parties)
    /// * `seed` - Random seed for this worker (should be unique per party)
    #[instrument(skip(transport), level = "info", fields(party_index = party_index))]
    pub fn new(
        config: PrivateTrainingConfig,
        transport: NodeTransport,
        party_index: usize,
        seed: u64,
    ) -> Self {
        let mpc_config = config.to_mpc_config();
        let trainer = MPCTrainer::new(mpc_config, transport, party_index, seed);

        info!(
            party_index = party_index,
            d_in = config.d_in,
            d_hid = config.d_hid,
            d_out = config.d_out,
            "Private training worker created"
        );

        Self {
            trainer,
            config,
            party_index,
            weights_shared: false,
            triples_generated: false,
        }
    }

    /// Phase 1: Share initial weights across all parties.
    ///
    /// Party 0 (dealer) generates additive shares and distributes them.
    /// Other parties receive their shares via the transport.
    ///
    /// # Arguments
    ///
    /// * `initial_weights` - The initial model weights (only used by party 0).
    ///   Pass `Some(weights)` for party 0, `None` for other parties.
    #[instrument(skip(self, initial_weights), level = "info", fields(party = self.party_index))]
    pub async fn share_weights(
        &mut self,
        initial_weights: Option<MpcModelWeights>,
    ) -> MPCResult<()> {
        if self.weights_shared {
            warn!(party = self.party_index, "Weights already shared, skipping");
            return Ok(());
        }

        self.trainer.share_weights(initial_weights).await?;
        self.weights_shared = true;

        info!(party = self.party_index, "Weight sharing complete");
        Ok(())
    }

    /// Phase 2: Generate Beaver triples for secure multiplication.
    ///
    /// Parties jointly generate triples via pairwise OT over the transport.
    /// The number of triples needed depends on model dimensions:
    /// - Each training step requires approximately `d_out * d_hid` triples
    ///   for the W2@h matmul plus `d_hid` for the ReLU backprop.
    #[instrument(skip(self), level = "info", fields(party = self.party_index, count = count))]
    pub async fn generate_beaver_triples(&mut self, count: usize) -> MPCResult<()> {
        if !self.weights_shared {
            return Err(MPCError::ProtocolError(
                "Cannot generate Beaver triples before weight sharing".into(),
            ));
        }

        self.trainer.generate_beaver_triples(count).await?;
        self.triples_generated = true;

        info!(
            party = self.party_index,
            count = count,
            remaining = self.trainer.beaver_triples_remaining(),
            "Beaver triple generation complete"
        );
        Ok(())
    }

    /// Phase 3-5: Run a single MPC training step.
    ///
    /// This performs:
    /// 1. Secure forward pass (matmul with Beaver triples, sign-bit ReLU)
    /// 2. Secure backward pass (gradients stay secret-shared)
    /// 3. Weight update (local per-share operation)
    /// 4. Optional re-sharing
    /// 5. Optional ZK proof generation (party 0 only)
    ///
    /// # Arguments
    ///
    /// * `input` - Training input vector (public, same for all parties)
    /// * `target` - Training target vector (public, same for all parties)
    #[instrument(skip(self, input, target), level = "debug", fields(
        party = self.party_index,
        step = self.trainer.current_step(),
    ))]
    pub async fn training_step(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<MPCTrainingStepResult> {
        if !self.weights_shared {
            return Err(MPCError::ProtocolError(
                "Cannot train before weight sharing".into(),
            ));
        }
        if !self.triples_generated {
            return Err(MPCError::ProtocolError(
                "Cannot train before generating Beaver triples".into(),
            ));
        }

        // Check if we have enough triples for this step.
        // Forward + backward + sign evaluation, with safety margin.
        let triples_per_step = (self.config.d_hid * self.config.d_in
            + self.config.d_out * self.config.d_hid
            + self.config.d_hid) * 4;
        let remaining = self.trainer.beaver_triples_remaining();
        if remaining < triples_per_step {
            warn!(
                party = self.party_index,
                remaining = remaining,
                needed = triples_per_step,
                "Insufficient Beaver triples, generating more"
            );
            self.trainer.generate_beaver_triples(triples_per_step * 2).await?;
        }

        self.trainer.training_step(input, target).await
    }

    /// Returns this worker's party index.
    pub fn party_index(&self) -> usize {
        self.party_index
    }

    /// Returns the current training step number.
    pub fn current_step(&self) -> u64 {
        self.trainer.current_step()
    }

    /// Returns the number of remaining Beaver triples.
    pub fn beaver_triples_remaining(&self) -> usize {
        self.trainer.beaver_triples_remaining()
    }

    /// Returns whether weight sharing has been completed.
    pub fn is_initialized(&self) -> bool {
        self.weights_shared && self.triples_generated
    }
}

// ──────────────────────────────────────────────────────────────
// Cleartext fallback
// ──────────────────────────────────────────────────────────────

/// Result of a cleartext training step (no MPC).
#[derive(Debug)]
pub struct CleartextStepResult {
    /// Training step number.
    pub step: u64,
    /// Computed loss.
    pub loss: f64,
    /// Updated model weights.
    pub model: MlpModel,
}

/// Runs a single training step in cleartext (no privacy).
///
/// Used as a fallback when MPC is disabled. Performs standard
/// forward/backward/SGD on the full model.
pub fn run_cleartext_training_step(
    model: &MlpModel,
    input: &[f64],
    target: &[f64],
    learning_rate: f64,
    step: u64,
) -> CleartextStepResult {
    use crate::trainer::{forward, backward, sgd_update};

    let fwd = forward(model, input, target);
    let grads = backward(model, input, target, &fwd);
    let mut updated = model.clone();
    sgd_update(&mut updated, &grads, learning_rate);

    CleartextStepResult {
        step,
        loss: fwd.loss,
        model: updated,
    }
}

// ──────────────────────────────────────────────────────────────
// High-level training round
// ──────────────────────────────────────────────────────────────

/// Result of a complete private training round.
#[derive(Debug)]
pub struct PrivateTrainingRoundResult {
    /// Session ID.
    pub session_id: String,
    /// Results from each training step (one per step, from party 0).
    pub step_results: Vec<StepSummary>,
    /// Whether training used MPC or cleartext.
    pub mode: TrainingMode,
    /// Total number of steps completed.
    pub total_steps: usize,
}

/// Summary of a single training step.
#[derive(Debug, Clone)]
pub struct StepSummary {
    /// Step number.
    pub step: u64,
    /// Loss value.
    pub loss: f64,
    /// Whether resharing occurred.
    pub reshared: bool,
    /// Whether a ZK proof was generated.
    pub has_proof: bool,
}

/// A single training sample (input, target).
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub input: Vec<f64>,
    pub target: Vec<f64>,
}

/// Runs a complete private training round.
///
/// This is the main entry point for MPC-backed training. It:
/// 1. Creates a `PrivateTrainingSession` with transport mesh
/// 2. Spawns one `PrivateTrainingWorker` per party as tokio tasks
/// 3. Coordinates weight sharing and Beaver triple generation
/// 4. Runs all training steps in lock-step across all parties
/// 5. Collects results from all workers
///
/// If `config.mode` is `Cleartext`, falls back to standard training
/// without MPC (useful for testing/benchmarking).
///
/// # Arguments
///
/// * `config` - Training configuration
/// * `model` - Initial model weights
/// * `samples` - Training data (input/target pairs)
/// * `session_id` - Unique session identifier
pub async fn run_private_training_round(
    config: PrivateTrainingConfig,
    model: &MlpModel,
    samples: &[TrainingSample],
    session_id: impl Into<String>,
) -> MPCResult<PrivateTrainingRoundResult> {
    let session_id_str: String = session_id.into();

    if config.mode == TrainingMode::Cleartext {
        return run_cleartext_round(config, model, samples, session_id_str);
    }

    // ── MPC mode ──
    let n = config.num_parties;
    let num_steps = samples.len();

    info!(
        session_id = %session_id_str,
        num_parties = n,
        num_steps = num_steps,
        "Starting private training round"
    );

    // Calculate Beaver triples needed for all steps.
    // Forward: W1@x (d_hid*d_in) + W2@h (d_out*d_hid) + ReLU sign (d_hid)
    // Backward: gradient multiplications (~same as forward)
    // Use 4x safety margin for multi-party overhead and resharing.
    let triples_per_step = (config.d_hid * config.d_in
        + config.d_out * config.d_hid
        + config.d_hid) * 4;
    let total_triples = triples_per_step * (num_steps + 1);

    // Create session and extract transports.
    let mut session = PrivateTrainingSession::new(config.clone(), model, &session_id_str)?;

    let initial_weights = session.initial_weights().clone();
    let mpc_config = config.clone();

    // Spawn workers as concurrent tokio tasks.
    let mut handles: Vec<JoinHandle<MPCResult<Vec<MPCTrainingStepResult>>>> = Vec::with_capacity(n);

    // Each worker needs its own copy of the samples.
    let samples_vec: Vec<TrainingSample> = samples.to_vec();

    for i in 0..n {
        let transport = session.take_worker_transport(i).ok_or_else(|| {
            MPCError::ProtocolError(format!("Transport for party {} already taken", i))
        })?;

        let worker_config = mpc_config.clone();
        let worker_weights = initial_weights.clone();
        let worker_samples = samples_vec.clone();
        let worker_total_triples = total_triples;

        handles.push(tokio::spawn(async move {
            let mut worker = PrivateTrainingWorker::new(
                worker_config,
                transport,
                i,
                42 + i as u64, // Unique seed per party
            );

            // Phase 1: Share weights (party 0 distributes, others receive).
            let weights_for_dealer = if i == 0 {
                Some(worker_weights)
            } else {
                None
            };
            worker.share_weights(weights_for_dealer).await?;

            // Phase 2: Generate Beaver triples.
            worker.generate_beaver_triples(worker_total_triples).await?;

            // Phase 3-5: Training steps.
            let mut results = Vec::with_capacity(worker_samples.len());
            for sample in &worker_samples {
                let result = worker.training_step(&sample.input, &sample.target).await?;
                results.push(result);
            }

            Ok(results)
        }));
    }

    // Wait for all workers to complete.
    let mut all_results: Vec<Vec<MPCTrainingStepResult>> = Vec::with_capacity(n);
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(results)) => {
                debug!(party = i, steps = results.len(), "Worker completed");
                all_results.push(results);
            }
            Ok(Err(e)) => {
                error!(party = i, error = %e, "Worker failed with MPC error");
                return Err(e);
            }
            Err(e) => {
                error!(party = i, error = %e, "Worker task panicked");
                return Err(MPCError::ProtocolError(format!(
                    "Worker {} panicked: {}", i, e
                )));
            }
        }
    }

    // Use party 0's results as the canonical results (they have the proofs).
    let party0_results = all_results.into_iter().next().ok_or_else(|| {
        MPCError::ProtocolError("No worker results".into())
    })?;

    let step_summaries: Vec<StepSummary> = party0_results
        .iter()
        .map(|r| StepSummary {
            step: r.step,
            loss: r.loss,
            reshared: r.reshared,
            has_proof: r.on_chain_proof.is_some(),
        })
        .collect();

    let total = step_summaries.len();

    info!(
        session_id = %session_id_str,
        total_steps = total,
        final_loss = step_summaries.last().map(|s| s.loss).unwrap_or(0.0),
        "Private training round complete"
    );

    Ok(PrivateTrainingRoundResult {
        session_id: session_id_str,
        step_results: step_summaries,
        mode: TrainingMode::Private,
        total_steps: total,
    })
}

/// Runs a complete training round in cleartext (no MPC).
fn run_cleartext_round(
    config: PrivateTrainingConfig,
    model: &MlpModel,
    samples: &[TrainingSample],
    session_id: String,
) -> MPCResult<PrivateTrainingRoundResult> {
    info!(
        session_id = %session_id,
        num_steps = samples.len(),
        "Running cleartext training round (no MPC)"
    );

    let mut current_model = model.clone();
    let mut step_summaries = Vec::with_capacity(samples.len());

    for (i, sample) in samples.iter().enumerate() {
        let result = run_cleartext_training_step(
            &current_model,
            &sample.input,
            &sample.target,
            config.learning_rate,
            i as u64,
        );

        step_summaries.push(StepSummary {
            step: i as u64,
            loss: result.loss,
            reshared: false,
            has_proof: false,
        });

        current_model = result.model;
    }

    let total = step_summaries.len();

    info!(
        session_id = %session_id,
        total_steps = total,
        final_loss = step_summaries.last().map(|s| s.loss).unwrap_or(0.0),
        "Cleartext training round complete"
    );

    Ok(PrivateTrainingRoundResult {
        session_id,
        step_results: step_summaries,
        mode: TrainingMode::Cleartext,
        total_steps: total,
    })
}

// ──────────────────────────────────────────────────────────────
// Aggregator-side MPC session coordinator
// ──────────────────────────────────────────────────────────────

/// Coordinates MPC training sessions from the aggregator's perspective.
///
/// Wraps `MpcSessionOrchestrator` and adds transport creation. When a training
/// round starts, the coordinator:
/// 1. Uses the orchestrator to select and register parties
/// 2. Creates a `PrivateTrainingSession` with transport mesh
/// 3. Distributes transports to workers
/// 4. Monitors session health
/// 5. Completes or fails the session when the round ends
pub struct PrivateTrainingCoordinator {
    /// Session orchestrator for party management.
    orchestrator: crate::training::mpc_session::MpcSessionOrchestrator,
    /// Current active training session (if any).
    active_session: Option<PrivateTrainingSession>,
    /// Training configuration.
    config: PrivateTrainingConfig,
}

impl PrivateTrainingCoordinator {
    /// Creates a new private training coordinator.
    pub fn new(
        config: PrivateTrainingConfig,
        checkpoint_dir: Option<std::path::PathBuf>,
    ) -> Self {
        let orchestrator = crate::training::mpc_session::MpcSessionOrchestrator::new(
            config.num_parties,
            checkpoint_dir,
        );

        Self {
            orchestrator,
            active_session: None,
            config,
        }
    }

    /// Registers a worker peer as an MPC party.
    pub fn register_worker(&mut self, peer_id: &crate::network::messages::PeerId) -> Option<usize> {
        self.orchestrator.register_worker(peer_id)
    }

    /// Deregisters a worker from the MPC party pool.
    pub fn deregister_worker(&mut self, peer_id: &crate::network::messages::PeerId) {
        self.orchestrator.deregister_worker(peer_id);
    }

    /// Records a heartbeat from a worker.
    pub fn record_heartbeat(&mut self, peer_id: &crate::network::messages::PeerId) {
        self.orchestrator.record_heartbeat(peer_id);
    }

    /// Returns true if enough parties are available to start a session.
    pub fn can_start_session(&self) -> bool {
        self.orchestrator.can_start_session()
    }

    /// Starts a new MPC training session for the given round.
    ///
    /// Creates transport mesh and prepares the session for training.
    /// Returns the session ID and party assignments.
    #[instrument(skip(self, model), level = "info", fields(round_id = round_id))]
    pub fn start_session(
        &mut self,
        round_id: u64,
        model: &MlpModel,
    ) -> MPCResult<Option<(String, Vec<(usize, String)>)>> {
        // Use the orchestrator to select parties and create session.
        let (session_id, party_assignments) = match self.orchestrator.start_session(round_id) {
            Some(result) => result,
            None => {
                debug!(round_id = round_id, "Cannot start MPC session: insufficient parties");
                return Ok(None);
            }
        };

        // Create the training session with transport mesh.
        let session = PrivateTrainingSession::new(
            self.config.clone(),
            model,
            &session_id,
        )?;

        self.active_session = Some(session);

        info!(
            session_id = %session_id,
            round_id = round_id,
            parties = party_assignments.len(),
            "MPC training session started with transport mesh"
        );

        Ok(Some((session_id, party_assignments)))
    }

    /// Takes the transport for a specific party from the active session.
    pub fn take_worker_transport(&mut self, party_index: usize) -> Option<NodeTransport> {
        self.active_session
            .as_mut()
            .and_then(|s| s.take_worker_transport(party_index))
    }

    /// Completes the active MPC session.
    pub fn complete_session(&mut self) {
        self.active_session = None;
        self.orchestrator.complete_session();
    }

    /// Fails the active MPC session with a reason.
    pub fn fail_session(&mut self, reason: &str) {
        self.active_session = None;
        self.orchestrator.fail_session(reason);
    }

    /// Runs health checks on the active session.
    pub fn health_check(&mut self) -> Vec<crate::training::mpc_session::OrchestratorHealthEvent> {
        self.orchestrator.health_check()
    }

    /// Returns a reference to the underlying orchestrator.
    pub fn orchestrator(&self) -> &crate::training::mpc_session::MpcSessionOrchestrator {
        &self.orchestrator
    }

    /// Returns the active session, if any.
    pub fn active_session(&self) -> Option<&PrivateTrainingSession> {
        self.active_session.as_ref()
    }

    /// Returns the training configuration.
    pub fn config(&self) -> &PrivateTrainingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_mode_default() {
        assert_eq!(TrainingMode::default(), TrainingMode::Private);
    }

    #[test]
    fn test_config_default() {
        let config = PrivateTrainingConfig::default();
        assert_eq!(config.mode, TrainingMode::Private);
        assert_eq!(config.num_parties, 3);
        assert_eq!(config.d_in, 4);
    }

    #[test]
    fn test_config_small() {
        let config = PrivateTrainingConfig::small(2);
        assert_eq!(config.num_parties, 2);
        assert_eq!(config.d_in, 2);
        assert_eq!(config.d_hid, 2);
        assert_eq!(config.d_out, 1);
    }

    #[test]
    fn test_config_to_mpc() {
        let config = PrivateTrainingConfig::small(3);
        let mpc = config.to_mpc_config();
        assert_eq!(mpc.d_in, 2);
        assert_eq!(mpc.d_hid, 2);
        assert_eq!(mpc.d_out, 1);
        assert_eq!(mpc.num_parties, 3);
    }

    #[test]
    fn test_model_conversion_roundtrip() {
        let model = MlpModel::new_random(2, 3, 1, 42);
        let mpc_weights = mlp_to_mpc_weights(&model);
        let recovered = mpc_to_mlp_model(&mpc_weights, 2, 3, 1);

        // Check roundtrip (approximate due to f64→Fr→f64).
        for (orig, recov) in model.w1.iter().zip(recovered.w1.iter()) {
            assert!((orig - recov).abs() < 1e-6, "w1 mismatch: {} vs {}", orig, recov);
        }
        for (orig, recov) in model.b1.iter().zip(recovered.b1.iter()) {
            assert!((orig - recov).abs() < 1e-6, "b1 mismatch: {} vs {}", orig, recov);
        }
    }

    #[test]
    fn test_session_creation() {
        let config = PrivateTrainingConfig::small(3);
        let model = MlpModel::new_random(2, 2, 1, 42);
        let session = PrivateTrainingSession::new(config, &model, "test-session").unwrap();

        assert_eq!(session.session_id(), "test-session");
        assert_eq!(session.num_parties(), 3);
        assert_eq!(session.available_transports(), 3);
    }

    #[test]
    fn test_session_take_transport() {
        let config = PrivateTrainingConfig::small(2);
        let model = MlpModel::new_random(2, 2, 1, 42);
        let mut session = PrivateTrainingSession::new(config, &model, "test").unwrap();

        assert!(session.take_worker_transport(0).is_some());
        assert!(session.take_worker_transport(0).is_none()); // Already taken
        assert!(session.take_worker_transport(1).is_some());
        assert_eq!(session.available_transports(), 0);
    }

    #[test]
    fn test_session_min_parties() {
        let config = PrivateTrainingConfig {
            num_parties: 1,
            ..PrivateTrainingConfig::small(1)
        };
        let model = MlpModel::new_random(2, 2, 1, 42);
        let result = PrivateTrainingSession::new(config, &model, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_cleartext_training_step() {
        let model = MlpModel::new_random(2, 2, 1, 42);
        let result = run_cleartext_training_step(
            &model,
            &[0.5, 0.3],
            &[1.0],
            0.01,
            0,
        );

        assert_eq!(result.step, 0);
        assert!(result.loss >= 0.0);
        assert_eq!(result.model.d_in, 2);
    }

    #[tokio::test]
    async fn test_cleartext_round() {
        let config = PrivateTrainingConfig {
            mode: TrainingMode::Cleartext,
            ..PrivateTrainingConfig::small(2)
        };
        let model = MlpModel::new_random(2, 2, 1, 42);
        let samples = vec![
            TrainingSample {
                input: vec![0.5, 0.3],
                target: vec![1.0],
            },
            TrainingSample {
                input: vec![0.1, 0.9],
                target: vec![0.0],
            },
        ];

        let result = run_private_training_round(config, &model, &samples, "test-cleartext").await.unwrap();

        assert_eq!(result.mode, TrainingMode::Cleartext);
        assert_eq!(result.total_steps, 2);
        assert_eq!(result.step_results.len(), 2);
    }

    #[tokio::test]
    async fn test_private_2party_training() {
        let config = PrivateTrainingConfig {
            mode: TrainingMode::Private,
            num_parties: 2,
            beaver_batch_size: 256,
            generate_proofs: false,
            ..PrivateTrainingConfig::small(2)
        };
        let model = MlpModel::new_random(2, 2, 1, 42);
        let samples = vec![
            TrainingSample {
                input: vec![0.5, 0.3],
                target: vec![1.0],
            },
        ];

        let result = run_private_training_round(config, &model, &samples, "test-private-2p").await.unwrap();

        assert_eq!(result.mode, TrainingMode::Private);
        assert_eq!(result.total_steps, 1);
        assert!(result.step_results[0].loss >= 0.0);
    }

    #[tokio::test]
    async fn test_private_3party_multi_step() {
        let config = PrivateTrainingConfig {
            mode: TrainingMode::Private,
            num_parties: 3,
            beaver_batch_size: 512,
            generate_proofs: false,
            ..PrivateTrainingConfig::small(3)
        };
        let model = MlpModel::new_random(2, 2, 1, 42);
        let samples = vec![
            TrainingSample { input: vec![0.5, 0.3], target: vec![1.0] },
            TrainingSample { input: vec![0.1, 0.9], target: vec![0.0] },
            TrainingSample { input: vec![0.8, 0.2], target: vec![1.0] },
        ];

        let result = run_private_training_round(config, &model, &samples, "test-private-3p").await.unwrap();

        assert_eq!(result.mode, TrainingMode::Private);
        assert_eq!(result.total_steps, 3);

        // Loss should generally decrease over training steps.
        // (Not guaranteed for every step, but trend should be downward.)
        let first_loss = result.step_results[0].loss;
        let last_loss = result.step_results[2].loss;
        info!("Loss: first={:.6}, last={:.6}", first_loss, last_loss);
    }

    #[tokio::test]
    async fn test_worker_lifecycle() {
        let config = PrivateTrainingConfig::small(2);
        let model = MlpModel::new_random(2, 2, 1, 42);
        let mut session = PrivateTrainingSession::new(config.clone(), &model, "lifecycle-test").unwrap();

        let t0 = session.take_worker_transport(0).unwrap();
        let t1 = session.take_worker_transport(1).unwrap();

        let initial_weights = session.initial_weights().clone();

        // Create workers.
        let mut w0 = PrivateTrainingWorker::new(config.clone(), t0, 0, 100);
        let mut w1 = PrivateTrainingWorker::new(config.clone(), t1, 1, 200);

        assert!(!w0.is_initialized());
        assert!(!w1.is_initialized());

        // Share weights (must be concurrent).
        let (r0, r1) = tokio::join!(
            w0.share_weights(Some(initial_weights)),
            w1.share_weights(None),
        );
        r0.unwrap();
        r1.unwrap();

        assert!(w0.is_initialized() == false); // triples not yet generated
        assert!(w1.is_initialized() == false);

        // Generate Beaver triples (must be concurrent).
        let (r0, r1) = tokio::join!(
            w0.generate_beaver_triples(64),
            w1.generate_beaver_triples(64),
        );
        r0.unwrap();
        r1.unwrap();

        assert!(w0.is_initialized());
        assert!(w1.is_initialized());
        assert!(w0.beaver_triples_remaining() >= 64);

        // Training step (must be concurrent).
        let input = vec![0.5, 0.3];
        let target = vec![1.0];
        let (r0, r1) = tokio::join!(
            w0.training_step(&input, &target),
            w1.training_step(&input, &target),
        );
        let s0 = r0.unwrap();
        let s1 = r1.unwrap();

        // Both parties should compute the same loss.
        assert!((s0.loss - s1.loss).abs() < 1e-6, "Loss mismatch: {} vs {}", s0.loss, s1.loss);
        assert_eq!(w0.current_step(), 1);
        assert_eq!(w1.current_step(), 1);
    }
}
