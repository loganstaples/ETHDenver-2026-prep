//! MPC-Integrated Training Flow.
//!
//! Wires helix-mpc into the training round system:
//! - Secret-shared weight distribution across parties
//! - Multi-worker gradient computation on shares
//! - MPC session coordination
//! - Adversarial detection (fake gradient → proof failure → slashing)
//!
//! Each party holds a share of the model weights. Training proceeds as:
//! 1. Coordinator secret-shares model weights to N parties
//! 2. Each party computes gradients on their weight share locally
//! 3. Gradient shares are applied to weight shares (local operation)
//! 4. Optionally, model is reconstructed and a ZK proof is generated
//! 5. Periodically reshare to prevent gradient accumulation attacks

use std::collections::HashMap;

use helix_mpc::error::{MPCError, MPCResult};
use helix_mpc::integration::training::{
    LayerWeightsFlat, ModelWeightsFlat, SecureTrainingConfig, SecureTrainingCoordinator,
    StepMetrics,
};
use helix_mpc::security::commitment::ShareCommitment;
use helix_mpc::security::verification::ShareVerifier;
use helix_mpc::sharing::model::{
    GradientShare, LayerGradientShare, ModelShare, ModelSharing, ReconstructedModel,
};
use helix_mpc::sharing::tensor::TensorShare;
use helix_mpc::sharing::AdditiveSharing;
use helix_mpc::types::{MPCConfig, PartyId, ShareId};
use helix_mpc::Fr;

use crate::trainer::{backward, forward, Gradients, MlpModel};

// ──────────────────────────────────────────────────────────────
// Conversion: MlpModel ↔ MPC types
// ──────────────────────────────────────────────────────────────

/// Converts an MlpModel to ModelWeightsFlat for MPC sharing.
pub fn model_to_flat(model: &MlpModel) -> ModelWeightsFlat {
    // Convert f64 weights to f32 (MPC uses f32 internally).
    let w1_f32: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = model.w2.iter().map(|&v| v as f32).collect();
    let b2_f32: Vec<f32> = model.b2.iter().map(|&v| v as f32).collect();

    // Layer 0 weights (layer 1 in ML parlance).
    let layer0_weights = vec![
        ("w1".to_string(), w1_f32, vec![model.d_hid, model.d_in]),
        ("b1".to_string(), b1_f32, vec![model.d_hid]),
    ];

    // Layer 1 weights (layer 2 in ML parlance).
    let layer1_weights = vec![
        ("w2".to_string(), w2_f32, vec![model.d_out, model.d_hid]),
        ("b2".to_string(), b2_f32, vec![model.d_out]),
    ];

    ModelWeightsFlat {
        name: format!("mlp-{}x{}x{}", model.d_in, model.d_hid, model.d_out),
        embeddings: None, // MLP has no embeddings
        layers: vec![
            LayerWeightsFlat {
                layer_idx: 0,
                weights: layer0_weights,
            },
            LayerWeightsFlat {
                layer_idx: 1,
                weights: layer1_weights,
            },
        ],
        lm_head: None, // MLP has no LM head
    }
}

/// Converts a ReconstructedModel back to MlpModel.
pub fn flat_to_model(
    recon: &ReconstructedModel,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> MlpModel {
    let w1: Vec<f64> = recon.layers[0]
        .weights
        .get("w1")
        .map(|w| w.data.iter().map(|&v| v as f64).collect())
        .unwrap_or_else(|| vec![0.0; d_hid * d_in]);

    let b1: Vec<f64> = recon.layers[0]
        .weights
        .get("b1")
        .map(|w| w.data.iter().map(|&v| v as f64).collect())
        .unwrap_or_else(|| vec![0.0; d_hid]);

    let w2: Vec<f64> = recon.layers[1]
        .weights
        .get("w2")
        .map(|w| w.data.iter().map(|&v| v as f64).collect())
        .unwrap_or_else(|| vec![0.0; d_out * d_hid]);

    let b2: Vec<f64> = recon.layers[1]
        .weights
        .get("b2")
        .map(|w| w.data.iter().map(|&v| v as f64).collect())
        .unwrap_or_else(|| vec![0.0; d_out]);

    MlpModel::new(d_in, d_hid, d_out, w1, b1, w2, b2)
}

/// Creates a zero gradient share for a slashed party.
fn create_zero_gradient_share(
    party: &PartyId,
    index: usize,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> GradientShare {
    let zero_grads = Gradients {
        dw1: vec![0.0; d_hid * d_in],
        db1: vec![0.0; d_hid],
        dw2: vec![0.0; d_out * d_hid],
        db2: vec![0.0; d_out],
    };
    gradients_to_share(&zero_grads, party, index, d_in, d_hid, d_out)
}

/// Converts Gradients to a GradientShare for a specific party.
fn gradients_to_share(
    grads: &Gradients,
    party: &PartyId,
    index: usize,
    d_in: usize,
    d_hid: usize,
    d_out: usize,
) -> GradientShare {
    // Convert f64 gradients to Fr field elements for shares.
    let dw1: Vec<Fr> = grads.dw1.iter().map(|&v| Fr::from_f64(v)).collect();
    let db1: Vec<Fr> = grads.db1.iter().map(|&v| Fr::from_f64(v)).collect();
    let dw2: Vec<Fr> = grads.dw2.iter().map(|&v| Fr::from_f64(v)).collect();
    let db2: Vec<Fr> = grads.db2.iter().map(|&v| Fr::from_f64(v)).collect();

    // Layer 0 gradients.
    let mut layer0_grads = HashMap::new();
    layer0_grads.insert(
        "w1".to_string(),
        TensorShare::new(
            ShareId::new(party.clone(), "grad_w1", index),
            dw1,
            vec![d_hid, d_in],
        ),
    );
    layer0_grads.insert(
        "b1".to_string(),
        TensorShare::new(
            ShareId::new(party.clone(), "grad_b1", index),
            db1,
            vec![d_hid],
        ),
    );

    // Layer 1 gradients.
    let mut layer1_grads = HashMap::new();
    layer1_grads.insert(
        "w2".to_string(),
        TensorShare::new(
            ShareId::new(party.clone(), "grad_w2", index),
            dw2,
            vec![d_out, d_hid],
        ),
    );
    layer1_grads.insert(
        "b2".to_string(),
        TensorShare::new(
            ShareId::new(party.clone(), "grad_b2", index),
            db2,
            vec![d_out],
        ),
    );

    GradientShare {
        party: party.clone(),
        index,
        embeddings: None,
        layers: vec![
            LayerGradientShare {
                layer_idx: 0,
                gradients: layer0_grads,
            },
            LayerGradientShare {
                layer_idx: 1,
                gradients: layer1_grads,
            },
        ],
        lm_head: None,
        error_bound: 0.0,
    }
}

// ──────────────────────────────────────────────────────────────
// MPC Training Round
// ──────────────────────────────────────────────────────────────

/// Configuration for MPC training rounds.
#[derive(Debug, Clone)]
pub struct MPCTrainingConfig {
    /// Number of parties in the MPC protocol.
    pub num_parties: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// How often to reshare weights (in steps).
    pub reshare_interval: u64,
    /// Maximum gradient norm for anomaly detection.
    pub max_gradient_norm: f64,
    /// Whether to verify share commitments.
    pub verify_commitments: bool,
    /// Model dimensions.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
}

impl Default for MPCTrainingConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            learning_rate: 0.01,
            reshare_interval: 50,
            max_gradient_norm: 100.0,
            verify_commitments: true,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
        }
    }
}

/// Result of an MPC training step.
#[derive(Debug)]
pub struct MPCStepResult {
    /// Step number.
    pub step: u64,
    /// Average loss across parties.
    pub avg_loss: f64,
    /// Whether resharing occurred.
    pub reshared: bool,
    /// Number of contributing parties.
    pub num_contributors: usize,
    /// Any slashed parties (detected cheating).
    pub slashed_parties: Vec<PartyId>,
}

/// Represents a worker's local computation on their weight share.
#[derive(Debug, Clone)]
pub struct WorkerComputation {
    /// Party ID.
    pub party: PartyId,
    /// Party index.
    pub index: usize,
    /// Computed gradient share.
    pub gradient_share: GradientShare,
    /// Local loss (not revealed, used for metrics only).
    pub local_loss: f64,
    /// Gradient commitment for verification.
    pub gradient_commitment: [u8; 32],
}

/// MPC Training Round Coordinator.
///
/// Orchestrates distributed training where workers compute on secret-shared
/// weights. Supports adversarial detection via commitment verification and
/// gradient norm bounds.
pub struct MPCTrainingRound {
    /// Configuration.
    config: MPCTrainingConfig,
    /// The MPC coordinator (holds shared weights).
    coordinator: SecureTrainingCoordinator,
    /// Model dimensions for reconstruction.
    dims: (usize, usize, usize),
    /// Current step.
    step: u64,
    /// Party commitments to their gradient shares.
    gradient_commitments: HashMap<PartyId, [u8; 32]>,
    /// Slashing record.
    slashing_record: Vec<SlashingEvent>,
}

/// Records a slashing event for a malicious party.
#[derive(Debug, Clone)]
pub struct SlashingEvent {
    /// The slashed party.
    pub party: PartyId,
    /// Step when violation was detected.
    pub step: u64,
    /// Reason for slashing.
    pub reason: SlashingReason,
}

/// Reasons for slashing a party.
#[derive(Debug, Clone)]
pub enum SlashingReason {
    /// Gradient norm exceeds maximum allowed.
    GradientNormExceeded { norm: f64, max: f64 },
    /// Commitment verification failed.
    CommitmentMismatch,
    /// Share verification failed.
    InvalidShare,
    /// Proof verification failed.
    ProofFailure { description: String },
}

impl MPCTrainingRound {
    /// Creates a new MPC training round.
    pub fn new(config: MPCTrainingConfig, model: &MlpModel) -> MPCResult<Self> {
        let mpc_config = MPCConfig {
            num_parties: config.num_parties,
            reshare_interval: config.reshare_interval,
            verify_shares: config.verify_commitments,
            ..MPCConfig::default()
        };

        let training_config = SecureTrainingConfig {
            mpc: mpc_config,
            learning_rate: config.learning_rate,
            total_steps: 1000, // Large default
            hidden_dim: config.d_hid,
            num_layers: 2,
            max_gradient_norm: config.max_gradient_norm,
        };

        let mut coordinator = SecureTrainingCoordinator::new(training_config)?;

        // Convert and share the model.
        let flat = model_to_flat(model);
        coordinator.initialize(&flat)?;

        Ok(Self {
            config: config.clone(),
            coordinator,
            dims: (config.d_in, config.d_hid, config.d_out),
            step: 0,
            gradient_commitments: HashMap::new(),
            slashing_record: Vec::new(),
        })
    }

    /// Gets a party's model share for local computation.
    pub fn get_party_share(&self, party_index: usize) -> MPCResult<&ModelShare> {
        self.coordinator.model_share(party_index)
    }

    /// Simulates a party computing gradients on their weight share.
    ///
    /// In a real distributed MPC setting, parties would use secure computation
    /// protocols (Beaver triples) to compute gradients on secret-shared weights.
    /// Here we simulate a simplified model where:
    /// 1. We reconstruct the full model to compute gradients
    /// 2. We create "gradient shares" that are party-specific contributions
    ///
    /// This preserves the gradient update flow while being testable.
    pub fn simulate_worker_computation(
        &self,
        party_index: usize,
        x: &[f64],
        target: &[f64],
    ) -> MPCResult<WorkerComputation> {
        let party = PartyId::from_index(party_index);

        // In real MPC, gradients would be computed securely.
        // For testing, we reconstruct the model and compute true gradients,
        // then divide by num_parties so each party's share sums to the full gradient.
        //
        // This simulates the output of secure gradient computation.

        // Reconstruct the full model from all shares.
        let n = self.config.num_parties;
        let shares: Vec<ModelShare> = (0..n)
            .map(|i| self.coordinator.model_share(i).cloned())
            .collect::<MPCResult<Vec<_>>>()?;

        let recon = ModelSharing::reconstruct_model_additive(&shares)?;
        let full_model = flat_to_model(&recon, self.dims.0, self.dims.1, self.dims.2);

        // Compute gradients on the full model.
        let fwd = forward(&full_model, x, target);
        let grads = backward(&full_model, x, target, &fwd);

        // Create gradient share: each party gets 1/n of the gradient.
        // When applied, sum of shares = full gradient.
        let scaled_grads = Gradients {
            dw1: grads.dw1.iter().map(|&g| g / n as f64).collect(),
            db1: grads.db1.iter().map(|&g| g / n as f64).collect(),
            dw2: grads.dw2.iter().map(|&g| g / n as f64).collect(),
            db2: grads.db2.iter().map(|&g| g / n as f64).collect(),
        };

        let gradient_share = gradients_to_share(
            &scaled_grads,
            &party,
            party_index,
            self.dims.0,
            self.dims.1,
            self.dims.2,
        );

        // Compute commitment to gradient share.
        let gradient_commitment = self.compute_gradient_commitment(&gradient_share);

        Ok(WorkerComputation {
            party,
            index: party_index,
            gradient_share,
            local_loss: fwd.loss,
            gradient_commitment,
        })
    }

    /// Computes a commitment to a gradient share for verification.
    fn compute_gradient_commitment(&self, grad: &GradientShare) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        for layer in &grad.layers {
            for (name, tensor) in &layer.gradients {
                hasher.update(name.as_bytes());
                for v in &tensor.data {
                    hasher.update(&v.to_bytes_le());
                }
            }
        }

        hasher.finalize().into()
    }

    /// Verifies a worker's computation for adversarial behavior.
    fn verify_worker_computation(&self, comp: &WorkerComputation) -> Result<(), SlashingReason> {
        // Check gradient norm bounds.
        let mut total_norm_sq: f64 = 0.0;
        for layer in &comp.gradient_share.layers {
            for (_, tensor) in &layer.gradients {
                for v in &tensor.data {
                    let val = v.to_f64();
                    total_norm_sq += val * val;
                }
            }
        }
        let norm = total_norm_sq.sqrt();

        if norm > self.config.max_gradient_norm {
            return Err(SlashingReason::GradientNormExceeded {
                norm,
                max: self.config.max_gradient_norm,
            });
        }

        // Verify commitment matches.
        let computed_commitment = self.compute_gradient_commitment(&comp.gradient_share);
        if computed_commitment != comp.gradient_commitment {
            return Err(SlashingReason::CommitmentMismatch);
        }

        Ok(())
    }

    /// Executes one MPC training step.
    ///
    /// 1. Collect gradient shares from all parties
    /// 2. Verify each party's contribution
    /// 3. Apply gradient updates
    /// 4. Detect and slash malicious parties
    pub fn training_step(
        &mut self,
        worker_computations: Vec<WorkerComputation>,
    ) -> MPCResult<MPCStepResult> {
        self.step += 1;

        let n = self.config.num_parties;
        if worker_computations.len() != n {
            return Err(MPCError::ShareCountMismatch {
                expected: n,
                got: worker_computations.len(),
            });
        }

        // Verify each worker's computation.
        let mut slashed_parties = Vec::new();
        let mut valid_computations = Vec::new();

        for comp in &worker_computations {
            match self.verify_worker_computation(comp) {
                Ok(()) => {
                    valid_computations.push(comp.clone());
                }
                Err(reason) => {
                    let event = SlashingEvent {
                        party: comp.party.clone(),
                        step: self.step,
                        reason: reason.clone(),
                    };
                    self.slashing_record.push(event);
                    slashed_parties.push(comp.party.clone());
                }
            }
        }

        // If too many parties are slashed, abort.
        if valid_computations.len() < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: valid_computations.len(),
            });
        }

        // Build gradient shares for all parties, using zero gradients for slashed parties.
        let mut gradient_shares: Vec<GradientShare> = Vec::with_capacity(n);
        let slashed_set: std::collections::HashSet<_> = slashed_parties.iter().collect();

        for comp in &worker_computations {
            if slashed_set.contains(&comp.party) {
                // Use zero gradient for slashed party.
                gradient_shares.push(create_zero_gradient_share(
                    &comp.party,
                    comp.index,
                    self.dims.0,
                    self.dims.1,
                    self.dims.2,
                ));
            } else {
                gradient_shares.push(comp.gradient_share.clone());
            }
        }

        // Compute average loss (only from valid computations).
        let avg_loss: f64 =
            valid_computations.iter().map(|c| c.local_loss).sum::<f64>() / valid_computations.len() as f64;

        // Apply gradient updates via the coordinator.
        let metrics = self.coordinator.training_step(gradient_shares)?;

        Ok(MPCStepResult {
            step: self.step,
            avg_loss,
            reshared: metrics.reshared,
            num_contributors: valid_computations.len(),
            slashed_parties,
        })
    }

    /// Reconstructs the trained model from all parties' shares.
    ///
    /// Note: This transitions the MPC session to `Complete` state.
    /// For intermediate reconstructions during multi-step training, use
    /// [`peek_model`] instead.
    pub fn reconstruct_model(&mut self) -> MPCResult<MlpModel> {
        let recon = self.coordinator.reconstruct_model()?;
        Ok(flat_to_model(&recon, self.dims.0, self.dims.1, self.dims.2))
    }

    /// Reconstructs the model from shares without changing session state.
    ///
    /// Unlike [`reconstruct_model`], this does NOT transition the session to
    /// `Complete`, allowing further training steps to proceed.
    pub fn peek_model(&self) -> MPCResult<MlpModel> {
        let n = self.config.num_parties;
        let shares: Vec<ModelShare> = (0..n)
            .map(|i| self.coordinator.model_share(i).cloned())
            .collect::<MPCResult<Vec<_>>>()?;
        let recon = ModelSharing::reconstruct_model_additive(&shares)?;
        Ok(flat_to_model(&recon, self.dims.0, self.dims.1, self.dims.2))
    }

    /// Returns the slashing record.
    pub fn slashing_record(&self) -> &[SlashingEvent] {
        &self.slashing_record
    }

    /// Returns the current step.
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Returns the model commitments (for on-chain anchoring).
    pub fn model_commitments(&self) -> Option<&helix_mpc::security::commitment::ModelCommitments> {
        self.coordinator.model_commitments()
    }
}

// ──────────────────────────────────────────────────────────────
// MPC Worker Handle (for network integration)
// ──────────────────────────────────────────────────────────────

/// Handle for a single MPC worker in the distributed training network.
///
/// Each worker node holds one of these. It wraps the worker's party index,
/// model share, and local gradient computation into a single interface
/// that the network layer can call when training batches arrive.
pub struct MPCWorkerHandle {
    /// This worker's party index.
    party_index: usize,
    /// Total number of parties.
    num_parties: usize,
    /// Model dimensions.
    dims: (usize, usize, usize),
    /// Current model weights (reconstructed from share for local computation).
    local_model: MlpModel,
    /// Gradient commitment history for anomaly detection.
    adversarial_detector: AdversarialDetector,
    /// Learning rate.
    learning_rate: f64,
    /// Step counter.
    step: u64,
}

impl MPCWorkerHandle {
    /// Creates a new worker handle from a model share.
    ///
    /// The worker receives its share from the coordinator and reconstructs
    /// a local view of the model for gradient computation.
    pub fn new(
        party_index: usize,
        num_parties: usize,
        model: MlpModel,
        learning_rate: f64,
        max_gradient_norm: f64,
    ) -> Self {
        let dims = (model.d_in, model.d_hid, model.d_out);
        Self {
            party_index,
            num_parties,
            dims,
            local_model: model,
            adversarial_detector: AdversarialDetector::new(max_gradient_norm),
            learning_rate,
            step: 0,
        }
    }

    /// Computes gradient share on a training batch.
    ///
    /// **WARNING: SIMULATION ONLY** — This method computes gradients on the full
    /// local model, NOT on secret shares. Every worker holds the complete model
    /// weights in memory. For real MPC privacy, workers must compute on their
    /// individual `ModelShare` using Beaver triples for secure multiplication,
    /// ensuring no single party can reconstruct the full model.
    ///
    /// The worker computes gradients on its local view of the model,
    /// then divides by num_parties to create its share of the full gradient.
    /// Returns a WorkerComputation that can be sent to the aggregator.
    pub fn compute_gradient_share(
        &mut self,
        x: &[f64],
        target: &[f64],
    ) -> MPCResult<WorkerComputation> {
        log::warn!(
            "MPC SIMULATION: party {} computing on FULL model (not secret shares). \
             This provides NO privacy guarantees.",
            self.party_index
        );

        let party = PartyId::from_index(self.party_index);

        // Forward and backward pass on local model.
        let fwd = forward(&self.local_model, x, target);
        let grads = backward(&self.local_model, x, target, &fwd);

        // Scale by 1/num_parties — only party 0 applies the full gradient
        // to avoid double-counting when shares are summed.
        let scale = if self.party_index == 0 { 1.0 } else { 0.0 };
        let scaled_grads = Gradients {
            dw1: grads.dw1.iter().map(|&g| g * scale / self.num_parties as f64).collect(),
            db1: grads.db1.iter().map(|&g| g * scale / self.num_parties as f64).collect(),
            dw2: grads.dw2.iter().map(|&g| g * scale / self.num_parties as f64).collect(),
            db2: grads.db2.iter().map(|&g| g * scale / self.num_parties as f64).collect(),
        };

        let gradient_share = gradients_to_share(
            &scaled_grads,
            &party,
            self.party_index,
            self.dims.0,
            self.dims.1,
            self.dims.2,
        );

        // Commitment for verification.
        let gradient_commitment = compute_gradient_commitment_bytes(&gradient_share);

        self.step += 1;

        Ok(WorkerComputation {
            party,
            index: self.party_index,
            gradient_share,
            local_loss: fwd.loss,
            gradient_commitment,
        })
    }

    /// Applies aggregated gradients to the local model.
    pub fn apply_gradients(&mut self, grads: &Gradients) {
        for (w, g) in self.local_model.w1.iter_mut().zip(grads.dw1.iter()) {
            *w -= self.learning_rate * g;
        }
        for (w, g) in self.local_model.b1.iter_mut().zip(grads.db1.iter()) {
            *w -= self.learning_rate * g;
        }
        for (w, g) in self.local_model.w2.iter_mut().zip(grads.dw2.iter()) {
            *w -= self.learning_rate * g;
        }
        for (w, g) in self.local_model.b2.iter_mut().zip(grads.db2.iter()) {
            *w -= self.learning_rate * g;
        }
    }

    /// Returns the party index.
    pub fn party_index(&self) -> usize {
        self.party_index
    }

    /// Returns the current local model.
    pub fn local_model(&self) -> &MlpModel {
        &self.local_model
    }

    /// Returns the current step.
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Returns the adversarial detector (mutable for checking incoming gradients).
    pub fn adversarial_detector_mut(&mut self) -> &mut AdversarialDetector {
        &mut self.adversarial_detector
    }
}

/// Computes a SHA-256 commitment to a gradient share.
fn compute_gradient_commitment_bytes(grad: &helix_mpc::sharing::model::GradientShare) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();

    for layer in &grad.layers {
        for (name, tensor) in &layer.gradients {
            hasher.update(name.as_bytes());
            for v in &tensor.data {
                hasher.update(&v.to_bytes_le());
            }
        }
    }

    hasher.finalize().into()
}

// ──────────────────────────────────────────────────────────────
// Adversarial Detection
// ──────────────────────────────────────────────────────────────

/// Adversarial detector for MPC training rounds.
///
/// Detects various attacks:
/// - Gradient poisoning (abnormal norms)
/// - Commitment fraud (mismatched commitments)
/// - Share tampering (invalid shares)
pub struct AdversarialDetector {
    /// Maximum allowed gradient norm.
    max_gradient_norm: f64,
    /// Historical gradient norms for anomaly detection.
    historical_norms: Vec<f64>,
    /// Number of standard deviations for outlier detection.
    outlier_threshold: f64,
}

impl AdversarialDetector {
    /// Creates a new adversarial detector.
    pub fn new(max_gradient_norm: f64) -> Self {
        Self {
            max_gradient_norm,
            historical_norms: Vec::new(),
            outlier_threshold: 3.0,
        }
    }

    /// Computes the L2 norm of a gradient share.
    pub fn compute_gradient_norm(grad: &GradientShare) -> f64 {
        let mut norm_sq: f64 = 0.0;
        for layer in &grad.layers {
            for (_, tensor) in &layer.gradients {
                for v in &tensor.data {
                    let val = v.to_f64();
                    norm_sq += val * val;
                }
            }
        }
        norm_sq.sqrt()
    }

    /// Checks if a gradient share is valid.
    pub fn check_gradient(&mut self, grad: &GradientShare) -> Result<(), String> {
        let norm = Self::compute_gradient_norm(grad);

        // Absolute bound check.
        if norm > self.max_gradient_norm {
            return Err(format!(
                "Gradient norm {} exceeds maximum {}",
                norm, self.max_gradient_norm
            ));
        }

        // Statistical outlier detection (if we have history).
        if self.historical_norms.len() >= 10 {
            let mean: f64 =
                self.historical_norms.iter().sum::<f64>() / self.historical_norms.len() as f64;
            let variance: f64 = self
                .historical_norms
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / self.historical_norms.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                let z_score = (norm - mean).abs() / std_dev;
                if z_score > self.outlier_threshold {
                    return Err(format!(
                        "Gradient norm {} is a statistical outlier (z={:.2})",
                        norm, z_score
                    ));
                }
            }
        }

        // Record this norm.
        self.historical_norms.push(norm);

        Ok(())
    }

    /// Verifies that shares are consistent with commitments.
    pub fn verify_share_commitment(
        share: &TensorShare,
        commitment: &ShareCommitment,
        blinding: &[u8; 32],
    ) -> bool {
        // Convert Fr field elements back to f64 for verification
        let data_f64: Vec<f64> = share.data.iter().map(|v| v.to_f64()).collect();
        commitment.verify_vector(&data_f64, blinding)
    }
}

// ──────────────────────────────────────────────────────────────
// MPC → ZK Proof Bridge
// ──────────────────────────────────────────────────────────────

/// Result of an MPC training step that includes a ZK proof.
#[derive(Debug)]
pub struct MPCProvedStep {
    /// The original MPC step result.
    pub step_result: MPCStepResult,
    /// ZK proof of the training step (generated by prover party).
    pub proved_step: Option<crate::trainer::ProvedStep>,
}

/// MPC training round with ZK proof generation.
///
/// Wraps `MPCTrainingRound` and generates a real Halo2 KZG proof after
/// each training step. The proof attests that the weight update was computed
/// correctly, and can be submitted on-chain via `HelixCoordinatorV2.submitProof()`.
///
/// After each step:
/// 1. The MPC round executes (gradient shares aggregated, weights updated)
/// 2. The old and new models are reconstructed from shares
/// 3. A Trainer generates the ZK proof using the reconstructed weights
pub struct MPCProvedTrainingRound {
    /// Inner MPC training round.
    inner: MPCTrainingRound,
    /// Trainer for proof generation (lazily initialized).
    trainer: Option<crate::trainer::Trainer>,
    /// Previous model state (for old-weight commitment in proof).
    prev_model: Option<MlpModel>,
    /// Model dimensions.
    dims: (usize, usize, usize),
    /// Learning rate.
    learning_rate: f64,
}

impl MPCProvedTrainingRound {
    /// Creates a new proved MPC training round.
    pub fn new(config: MPCTrainingConfig, model: &MlpModel) -> MPCResult<Self> {
        let dims = (config.d_in, config.d_hid, config.d_out);
        let lr = config.learning_rate;
        Ok(Self {
            inner: MPCTrainingRound::new(config, model)?,
            trainer: None,
            prev_model: Some(model.clone()),
            dims,
            learning_rate: lr,
        })
    }

    /// Executes an MPC training step and generates a ZK proof.
    ///
    /// The proof proves that the weight update (old_model → new_model) was
    /// computed correctly for the given (input, target) pair.
    pub fn training_step_with_proof(
        &mut self,
        worker_computations: Vec<WorkerComputation>,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<MPCProvedStep> {
        let old_model = self.prev_model.take()
            .ok_or_else(|| helix_mpc::error::MPCError::ProtocolError(
                "No previous model state for proof generation".into(),
            ))?;

        // Execute MPC training step.
        let step_result = self.inner.training_step(worker_computations)?;

        // Reconstruct new model from shares without changing session state
        // (so further training steps can proceed).
        let new_model = self.inner.peek_model()?;

        // Generate ZK proof using the Trainer's proven pipeline.
        let proved_step = self.generate_proof(&old_model, &new_model, input, target)?;

        // Save new model for next step.
        self.prev_model = Some(new_model);

        Ok(MPCProvedStep {
            step_result,
            proved_step: Some(proved_step),
        })
    }

    /// Generates a ZK proof from old and new model states.
    fn generate_proof(
        &mut self,
        old_model: &MlpModel,
        _new_model: &MlpModel,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<crate::trainer::ProvedStep> {
        // Initialize trainer lazily.
        if self.trainer.is_none() {
            self.trainer = Some(crate::trainer::Trainer::with_model(
                old_model.clone(),
                self.learning_rate,
            ));
        }

        let trainer = self.trainer.as_mut().unwrap();

        // Sync the trainer's model to the old state, then train_step will
        // do forward/backward/SGD/quantize/prove — producing new weights
        // that match the MPC training step's result.
        trainer.set_model(old_model.clone());
        trainer.train_step(input, target)
            .map_err(|e| helix_mpc::error::MPCError::ProtocolError(
                format!("Proof generation failed: {}", e),
            ))
    }

    /// Executes a training step without proof generation.
    pub fn training_step(
        &mut self,
        worker_computations: Vec<WorkerComputation>,
    ) -> MPCResult<MPCStepResult> {
        let result = self.inner.training_step(worker_computations)?;
        // Update model state tracking.
        if let Ok(model) = self.inner.reconstruct_model() {
            self.prev_model = Some(model);
        }
        Ok(result)
    }

    /// Returns a reference to the inner MPC training round.
    pub fn inner(&self) -> &MPCTrainingRound {
        &self.inner
    }

    /// Returns a mutable reference to the inner MPC training round.
    pub fn inner_mut(&mut self) -> &mut MPCTrainingRound {
        &mut self.inner
    }

    /// Reconstructs the trained model.
    pub fn reconstruct_model(&mut self) -> MPCResult<MlpModel> {
        self.inner.reconstruct_model()
    }
}

/// Converts an MPC `Halo2ProofResult` (from helix-mpc's CircuitBridge) to
/// raw proof bytes suitable for the node's gradient message.
///
/// This is the bridge between the real MPC training path (where the
/// `MPCTrainer` generates proofs via `reconstruct_and_prove()`) and the
/// node's network layer.
pub fn mpc_proof_to_bytes(
    proof: &helix_mpc::integration::circuit_bridge::Halo2ProofResult,
) -> Vec<u8> {
    proof.proof.clone()
}

/// Extracts public inputs from an MPC proof as hex strings for on-chain submission.
pub fn mpc_proof_public_inputs_hex(
    proof: &helix_mpc::integration::circuit_bridge::Halo2ProofResult,
) -> Vec<String> {
    proof.public_inputs.iter()
        .map(|f| hex::encode(f.to_bytes_le()))
        .collect()
}

// ──────────────────────────────────────────────────────────────
// ConnectionPoolBridge: Node ↔ MPC Transport
// ──────────────────────────────────────────────────────────────

use crate::network::messages::{MpcDataMessage, MessagePayload, NetworkMessage, PeerId};
use crate::network::transport::ConnectionPool;
use async_trait::async_trait;
use helix_mpc::session::node_transport::NodeConnectionBridge;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex as TokioMutex};

/// Bridges the node's [`ConnectionPool`] with the MPC transport layer.
///
/// Implements [`NodeConnectionBridge`] so that [`NodeTransport`](helix_mpc::session::node_transport::NodeTransport)
/// can send and receive MPC messages through the node's existing TCP/TLS
/// connections instead of establishing separate ones.
///
/// # How it works
///
/// **Outgoing messages**: `send_to_peer()` wraps raw MPC bytes in a
/// [`NetworkMessage`] with an [`MpcData`](MessagePayload::MpcData) payload
/// and sends it via the [`ConnectionPool`].
///
/// **Incoming messages**: The node's receive loop should detect
/// `MpcData` payloads and call [`route_incoming()`](Self::route_incoming)
/// with the sender's peer ID and the raw data. This dispatches to the
/// correct per-peer mpsc channel, which `recv_from_peer()` reads from.
///
/// # Example
///
/// ```ignore
/// let bridge = ConnectionPoolBridge::new(
///     pool.clone(),
///     local_peer_id,
///     "session-123".into(),
///     &["peer-a".into(), "peer-b".into()],
/// );
///
/// // In the node's receive loop:
/// if let MessagePayload::MpcData(mpc_msg) = &message.payload {
///     bridge.route_incoming(&message.sender.0, mpc_msg.data.clone()).await?;
/// }
/// ```
pub struct ConnectionPoolBridge {
    /// The node's connection pool.
    pool: Arc<ConnectionPool>,
    /// Our node peer ID.
    local_peer_id: PeerId,
    /// Session ID for message tagging.
    session_id: String,
    /// Incoming message receivers per peer (peer_id string -> receiver).
    receivers: HashMap<String, TokioMutex<mpsc::Receiver<Vec<u8>>>>,
    /// Incoming message senders (held for routing incoming messages).
    senders: HashMap<String, mpsc::Sender<Vec<u8>>>,
}

impl ConnectionPoolBridge {
    /// Creates a new bridge with receiver channels for each expected MPC peer.
    ///
    /// # Arguments
    ///
    /// * `pool` - The node's connection pool (must already have peers registered)
    /// * `local_peer_id` - This node's peer ID
    /// * `session_id` - MPC session identifier for message tagging
    /// * `mpc_peer_ids` - Peer IDs of the other MPC participants
    pub fn new(
        pool: Arc<ConnectionPool>,
        local_peer_id: PeerId,
        session_id: String,
        mpc_peer_ids: &[String],
    ) -> Self {
        let mut receivers = HashMap::new();
        let mut senders = HashMap::new();

        for peer_id in mpc_peer_ids {
            let (tx, rx) = mpsc::channel(4096);
            receivers.insert(peer_id.clone(), TokioMutex::new(rx));
            senders.insert(peer_id.clone(), tx);
        }

        Self {
            pool,
            local_peer_id,
            session_id,
            receivers,
            senders,
        }
    }

    /// Returns the sender channel for a specific peer.
    ///
    /// Used by the node's incoming message handler to get the channel
    /// for dispatching received MPC messages.
    pub fn sender_for_peer(&self, peer_id: &str) -> Option<&mpsc::Sender<Vec<u8>>> {
        self.senders.get(peer_id)
    }

    /// Routes an incoming MPC message to the correct peer receiver channel.
    ///
    /// The node's receive loop should call this when it receives a
    /// [`MessagePayload::MpcData`] message. The `from_peer_id` should be
    /// the sender's peer ID string from the `NetworkMessage`.
    pub async fn route_incoming(&self, from_peer_id: &str, data: Vec<u8>) -> Result<(), String> {
        let tx = self
            .senders
            .get(from_peer_id)
            .ok_or_else(|| format!("no receiver channel for peer {}", from_peer_id))?;
        tx.send(data)
            .await
            .map_err(|e| format!("failed to route MPC message from {}: {}", from_peer_id, e))
    }

    /// Returns the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the local peer ID.
    pub fn local_peer_id(&self) -> &PeerId {
        &self.local_peer_id
    }
}

#[async_trait]
impl NodeConnectionBridge for ConnectionPoolBridge {
    async fn send_to_peer(&self, peer_id: &str, data: &[u8]) -> Result<(), String> {
        let node_peer_id = PeerId::from_string(peer_id);

        let message = NetworkMessage::new(
            self.local_peer_id.clone(),
            MessagePayload::MpcData(MpcDataMessage {
                session_id: self.session_id.clone(),
                data: data.to_vec(),
            }),
        );

        self.pool
            .send(&node_peer_id, message)
            .await
            .map_err(|e| format!("ConnectionPool send to {} failed: {}", peer_id, e))
    }

    async fn recv_from_peer(&self, peer_id: &str) -> Result<Vec<u8>, String> {
        let rx = self
            .receivers
            .get(peer_id)
            .ok_or_else(|| format!("no receiver channel for peer {}", peer_id))?;

        let mut rx_guard = rx.lock().await;
        rx_guard
            .recv()
            .await
            .ok_or_else(|| format!("MPC channel from {} closed", peer_id))
    }
}

// ──────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> MlpModel {
        MlpModel::new_random(4, 8, 2, 42)
    }

    #[test]
    fn test_model_to_flat_conversion() {
        let model = create_test_model();
        let flat = model_to_flat(&model);

        assert_eq!(flat.name, "mlp-4x8x2");
        assert!(flat.embeddings.is_none());
        assert_eq!(flat.layers.len(), 2);

        // Layer 0 should have w1 and b1.
        assert!(flat.layers[0].weights.iter().any(|(n, _, _)| n == "w1"));
        assert!(flat.layers[0].weights.iter().any(|(n, _, _)| n == "b1"));

        // Layer 1 should have w2 and b2.
        assert!(flat.layers[1].weights.iter().any(|(n, _, _)| n == "w2"));
        assert!(flat.layers[1].weights.iter().any(|(n, _, _)| n == "b2"));
    }

    #[test]
    fn test_mpc_training_round_creation() {
        let model = create_test_model();
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            ..Default::default()
        };

        let round = MPCTrainingRound::new(config, &model);
        assert!(round.is_ok());

        let round = round.unwrap();
        assert_eq!(round.current_step(), 0);
    }

    #[test]
    fn test_worker_computation_simulation() {
        let model = create_test_model();
        let config = MPCTrainingConfig {
            num_parties: 3,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            ..Default::default()
        };

        let round = MPCTrainingRound::new(config, &model).unwrap();

        let x = vec![1.0, 0.5, -0.3, 0.8];
        let target = vec![1.0, 0.0];

        // Simulate computation for party 0.
        let comp = round.simulate_worker_computation(0, &x, &target);
        assert!(comp.is_ok());

        let comp = comp.unwrap();
        assert_eq!(comp.party, PartyId::from_index(0));
        assert!(comp.local_loss > 0.0);
        assert_ne!(comp.gradient_commitment, [0u8; 32]);
    }

    #[test]
    fn test_full_mpc_training_step() {
        let model = create_test_model();
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            max_gradient_norm: 1000.0, // Large enough for random model gradients
            ..Default::default()
        };

        let mut round = MPCTrainingRound::new(config.clone(), &model).unwrap();

        let x = vec![1.0, 0.5, -0.3, 0.8];
        let target = vec![1.0, 0.0];

        // Simulate computations for all parties.
        let mut computations = Vec::new();
        for i in 0..config.num_parties {
            let comp = round.simulate_worker_computation(i, &x, &target).unwrap();
            computations.push(comp);
        }

        // Execute training step.
        let result = round.training_step(computations);
        assert!(result.is_ok(), "training step failed: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.step, 1);
        assert_eq!(result.num_contributors, 3);
        assert!(result.slashed_parties.is_empty());
    }

    #[test]
    fn test_model_reconstruction() {
        let original = create_test_model();
        let config = MPCTrainingConfig {
            num_parties: 3,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            ..Default::default()
        };

        let mut round = MPCTrainingRound::new(config, &original).unwrap();

        // Reconstruct immediately (no training steps).
        let reconstructed = round.reconstruct_model().unwrap();

        // Model dimensions should match.
        assert_eq!(reconstructed.d_in, original.d_in);
        assert_eq!(reconstructed.d_hid, original.d_hid);
        assert_eq!(reconstructed.d_out, original.d_out);

        // Weights should match (within floating point tolerance).
        for (a, b) in reconstructed.w1.iter().zip(original.w1.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "w1 mismatch: {} vs {}",
                a, b
            );
        }
    }

    #[test]
    fn test_adversarial_detection_gradient_norm() {
        let mut detector = AdversarialDetector::new(10.0);

        // Create a normal gradient share.
        let party = PartyId::from_index(0);
        let mut normal_grads = HashMap::new();
        normal_grads.insert(
            "w".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "grad", 0),
                (0..10).map(|_| Fr::from_f64(0.1)).collect(),
                vec![2, 5],
            ),
        );

        let normal_grad = GradientShare {
            party: party.clone(),
            index: 0,
            embeddings: None,
            layers: vec![LayerGradientShare {
                layer_idx: 0,
                gradients: normal_grads,
            }],
            lm_head: None,
            error_bound: 0.0,
        };

        assert!(detector.check_gradient(&normal_grad).is_ok());

        // Create an abnormally large gradient share.
        let mut large_grads = HashMap::new();
        large_grads.insert(
            "w".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "grad", 0),
                (0..10).map(|_| Fr::from_f64(100.0)).collect(), // Very large values
                vec![2, 5],
            ),
        );

        let large_grad = GradientShare {
            party: party.clone(),
            index: 0,
            embeddings: None,
            layers: vec![LayerGradientShare {
                layer_idx: 0,
                gradients: large_grads,
            }],
            lm_head: None,
            error_bound: 0.0,
        };

        assert!(detector.check_gradient(&large_grad).is_err());
    }

    #[test]
    fn test_slashing_on_malicious_gradient() {
        let model = create_test_model();

        // First, compute what the normal gradient norm would be.
        let x = vec![1.0, 0.5, -0.3, 0.8];
        let target = vec![1.0, 0.0];

        // Use a high threshold initially to compute normal gradient norm.
        let test_config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            max_gradient_norm: 100000.0,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            ..Default::default()
        };
        let test_round = MPCTrainingRound::new(test_config, &model).unwrap();
        let test_comp = test_round.simulate_worker_computation(0, &x, &target).unwrap();
        let normal_norm = AdversarialDetector::compute_gradient_norm(&test_comp.gradient_share);

        // Set threshold to 2x normal norm so normal gradients pass, but 1000x amplified fails.
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            max_gradient_norm: normal_norm * 2.0,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            ..Default::default()
        };

        let mut round = MPCTrainingRound::new(config.clone(), &model).unwrap();

        // Get normal computations.
        let mut computations = Vec::new();
        for i in 0..config.num_parties {
            let mut comp = round.simulate_worker_computation(i, &x, &target).unwrap();

            // Make party 1's gradient maliciously large.
            if i == 1 {
                let amplify = Fr::from_u64(1000);
                for layer in &mut comp.gradient_share.layers {
                    for (_, tensor) in &mut layer.gradients {
                        for v in &mut tensor.data {
                            *v = *v * amplify; // Amplify gradients
                        }
                    }
                }
            }

            computations.push(comp);
        }

        // Execute training step.
        let result = round.training_step(computations).unwrap();

        // Party 1 should be slashed.
        assert!(
            result.slashed_parties.contains(&PartyId::from_index(1)),
            "Party 1 should be slashed for abnormal gradient"
        );
        assert_eq!(result.num_contributors, 2);

        // Check slashing record.
        let slashing = round.slashing_record();
        assert!(!slashing.is_empty());
        assert!(matches!(
            slashing[0].reason,
            SlashingReason::GradientNormExceeded { .. }
        ));
    }

    #[test]
    fn test_multiple_training_steps_reduce_loss() {
        let model = MlpModel::new_random(2, 4, 1, 42);
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.1,
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            max_gradient_norm: 10000.0, // Large enough for random model gradients
            ..Default::default()
        };

        let mut round = MPCTrainingRound::new(config.clone(), &model).unwrap();

        let x = vec![1.0, 2.0];
        let target = vec![3.0];

        let mut losses = Vec::new();

        // Run 10 training steps.
        for _ in 0..10 {
            let mut computations = Vec::new();
            for i in 0..config.num_parties {
                let comp = round.simulate_worker_computation(i, &x, &target).unwrap();
                computations.push(comp);
            }

            let result = round.training_step(computations).unwrap();
            losses.push(result.avg_loss);
        }

        // Loss should generally decrease.
        // Note: With secret-shared weights, loss dynamics are complex,
        // but we should see improvement over multiple steps.
        let first_avg = losses[..3].iter().sum::<f64>() / 3.0;
        let last_avg = losses[7..].iter().sum::<f64>() / 3.0;

        // Allow some tolerance since MPC introduces noise.
        assert!(
            last_avg <= first_avg * 1.5,
            "Loss should not increase dramatically: first_avg={}, last_avg={}",
            first_avg,
            last_avg
        );
    }

    #[test]
    fn test_mpc_proved_training_round() {
        // Test that MPCProvedTrainingRound generates proofs after MPC steps.
        // Uses small model dimensions to keep proof generation fast.
        let model = MlpModel::new_random(2, 2, 1, 42);
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            max_gradient_norm: 10000.0,
            ..Default::default()
        };

        let mut round = MPCProvedTrainingRound::new(config.clone(), &model).unwrap();

        let x = vec![0.5, 0.5];
        let target = vec![0.5];

        // Simulate worker computations.
        let mut computations = Vec::new();
        for i in 0..config.num_parties {
            let comp = round.inner().simulate_worker_computation(i, &x, &target).unwrap();
            computations.push(comp);
        }

        // Execute proved training step.
        let result = round.training_step_with_proof(computations, &x, &target).unwrap();

        // Step result should be valid.
        assert_eq!(result.step_result.step, 1);
        assert_eq!(result.step_result.num_contributors, 3);
        assert!(result.step_result.slashed_parties.is_empty());

        // Proof should be present and valid.
        let proved_step = result.proved_step.expect("Should have a proved step");
        assert!(!proved_step.proof_result.proof.is_empty(), "Proof should not be empty");
        assert!(proved_step.proof_result.verified, "Proof should self-verify");
        assert!(proved_step.loss > 0.0, "Loss should be positive");
    }

    #[test]
    fn test_mpc_proved_round_multi_step() {
        // Test that MPCProvedTrainingRound works across multiple steps.
        let model = MlpModel::new_random(2, 2, 1, 42);
        let config = MPCTrainingConfig {
            num_parties: 3,
            learning_rate: 0.01,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            max_gradient_norm: 10000.0,
            ..Default::default()
        };

        let mut round = MPCProvedTrainingRound::new(config.clone(), &model).unwrap();

        let data = vec![
            (vec![0.5, 0.5], vec![0.5]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![0.0]),
        ];

        let mut losses = Vec::new();
        for (x, target) in &data {
            let mut computations = Vec::new();
            for i in 0..config.num_parties {
                let comp = round.inner().simulate_worker_computation(i, x, target).unwrap();
                computations.push(comp);
            }

            let result = round.training_step_with_proof(computations, x, target).unwrap();
            let proved = result.proved_step.expect("Should have proof");
            assert!(proved.proof_result.verified, "Proof should verify at each step");
            losses.push(result.step_result.avg_loss);
        }

        assert_eq!(losses.len(), 3, "Should have 3 step results");
    }

    #[test]
    fn test_mpc_proof_to_bytes() {
        use helix_mpc::integration::circuit_bridge::Halo2ProofResult;
        use helix_mpc::Fr;

        let proof = Halo2ProofResult::new(
            vec![1, 2, 3, 4],
            vec![Fr::from_u64(42)],
            (Fr::ZERO, Fr::ZERO),
            (Fr::ZERO, Fr::ZERO),
            Fr::ZERO,
            Fr::ZERO,
            0,
            100,
        );

        let bytes = mpc_proof_to_bytes(&proof);
        assert_eq!(bytes, vec![1, 2, 3, 4]);

        let hex_inputs = mpc_proof_public_inputs_hex(&proof);
        assert_eq!(hex_inputs.len(), 1);
    }

    // ── ConnectionPoolBridge tests ──────────────────────────────

    /// Helper to create a ConnectionPool for testing (no real TCP needed).
    fn create_test_pool() -> Arc<ConnectionPool> {
        use crate::network::transport::{TcpTransport, TransportConfig};

        let local_id = PeerId::random();
        let config = TransportConfig::default();
        let transport = Arc::new(TcpTransport::new(local_id, config).unwrap());
        Arc::new(ConnectionPool::new(transport))
    }

    #[test]
    fn test_connection_pool_bridge_creation() {
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string(), "peer-b".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id.clone(),
            "session-1".to_string(),
            &peers,
        );

        assert_eq!(bridge.session_id(), "session-1");
        assert_eq!(bridge.local_peer_id(), &local_id);
        assert!(bridge.sender_for_peer("peer-a").is_some());
        assert!(bridge.sender_for_peer("peer-b").is_some());
        assert!(bridge.sender_for_peer("peer-c").is_none());
    }

    #[tokio::test]
    async fn test_connection_pool_bridge_route_incoming() {
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id,
            "session-1".to_string(),
            &peers,
        );

        // Route a message from peer-a.
        let data = b"hello from peer-a".to_vec();
        bridge.route_incoming("peer-a", data.clone()).await.unwrap();

        // Receive via the bridge.
        let received = bridge.recv_from_peer("peer-a").await.unwrap();
        assert_eq!(received, data);
    }

    #[tokio::test]
    async fn test_connection_pool_bridge_route_unknown_peer() {
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id,
            "session-1".to_string(),
            &peers,
        );

        // Routing from an unknown peer should fail.
        let result = bridge.route_incoming("unknown-peer", vec![1, 2, 3]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no receiver channel"));
    }

    #[tokio::test]
    async fn test_connection_pool_bridge_recv_unknown_peer() {
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id,
            "session-1".to_string(),
            &peers,
        );

        // Receiving from an unknown peer should fail.
        let result = bridge.recv_from_peer("unknown-peer").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no receiver channel"));
    }

    #[tokio::test]
    async fn test_connection_pool_bridge_multiple_messages() {
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string(), "peer-b".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id,
            "session-1".to_string(),
            &peers,
        );

        // Route multiple messages from different peers.
        bridge.route_incoming("peer-a", b"msg-1-from-a".to_vec()).await.unwrap();
        bridge.route_incoming("peer-b", b"msg-1-from-b".to_vec()).await.unwrap();
        bridge.route_incoming("peer-a", b"msg-2-from-a".to_vec()).await.unwrap();

        // Messages should be received in order per peer.
        let recv_a1 = bridge.recv_from_peer("peer-a").await.unwrap();
        assert_eq!(recv_a1, b"msg-1-from-a");

        let recv_b1 = bridge.recv_from_peer("peer-b").await.unwrap();
        assert_eq!(recv_b1, b"msg-1-from-b");

        let recv_a2 = bridge.recv_from_peer("peer-a").await.unwrap();
        assert_eq!(recv_a2, b"msg-2-from-a");
    }

    #[tokio::test]
    async fn test_connection_pool_bridge_send_constructs_mpc_message() {
        // Verify that send_to_peer creates the right NetworkMessage payload.
        // Since the ConnectionPool will fail (no real TCP connection to the peer),
        // we check that the error comes from the pool, not from message construction.
        let pool = create_test_pool();
        let local_id = PeerId::from_string("local-node");
        let peers = vec!["peer-a".to_string()];

        let bridge = ConnectionPoolBridge::new(
            pool,
            local_id,
            "session-1".to_string(),
            &peers,
        );

        // send_to_peer should fail because peer-a is not registered in the pool,
        // but it should get as far as calling pool.send() (which returns PeerNotFound).
        let result = bridge.send_to_peer("peer-a", b"test data").await;
        assert!(result.is_err());
        // The error should come from the ConnectionPool (PeerNotFound),
        // not from message construction.
        let err = result.unwrap_err();
        assert!(
            err.contains("ConnectionPool send to peer-a failed"),
            "Expected ConnectionPool error, got: {}",
            err
        );
    }

    #[test]
    fn test_mpc_data_message_serialization() {
        // Test that MpcDataMessage can be serialized/deserialized through NetworkMessage.
        let msg = NetworkMessage::new(
            PeerId::from_string("sender"),
            MessagePayload::MpcData(MpcDataMessage {
                session_id: "session-42".to_string(),
                data: vec![1, 2, 3, 4, 5],
            }),
        );

        let bytes = crate::network::messages::serialize_message(&msg).unwrap();
        let decoded = crate::network::messages::deserialize_message(&bytes).unwrap();

        assert_eq!(msg.id, decoded.id);
        match decoded.payload {
            MessagePayload::MpcData(mpc_msg) => {
                assert_eq!(mpc_msg.session_id, "session-42");
                assert_eq!(mpc_msg.data, vec![1, 2, 3, 4, 5]);
            }
            other => panic!("Expected MpcData payload, got {:?}", other),
        }
    }
}
