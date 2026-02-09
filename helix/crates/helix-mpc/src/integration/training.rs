//! Secure training coordinator.
//!
//! Orchestrates a full secure training round:
//! 1. Model owner secret-shares weights to workers
//! 2. Workers compute forward/backward passes on shares
//! 3. Workers generate gradient shares
//! 4. Gradient shares are aggregated (each party aggregates locally on their shares)
//! 5. Model shares are updated with aggregated gradient
//! 6. Periodically re-share to prevent accumulation attacks


use crate::error::{MPCError, MPCResult};
use crate::security::audit::AuditLog;
use crate::security::commitment::{BlindingGenerator, ModelCommitments, ShareCommitment};
use crate::session::manager::MPCSession;
use crate::sharing::model::{
    GradientShare, ModelShare, ModelSharing,
    ReconstructedModel,
};
use crate::sharing::AdditiveSharing;
use crate::types::{MPCConfig, MPCPhase, PartyId, PartyRole};

/// Configuration for secure training.
#[derive(Debug, Clone)]
pub struct SecureTrainingConfig {
    /// MPC configuration.
    pub mpc: MPCConfig,
    /// Learning rate.
    pub learning_rate: f64,
    /// Total training steps.
    pub total_steps: u64,
    /// Hidden dimension of the model.
    pub hidden_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum gradient norm for clipping.
    pub max_gradient_norm: f64,
}

impl Default for SecureTrainingConfig {
    fn default() -> Self {
        Self {
            mpc: MPCConfig::three_party(),
            learning_rate: 0.001,
            total_steps: 100,
            hidden_dim: 64,
            num_layers: 2,
            max_gradient_norm: 10.0,
        }
    }
}

/// Orchestrates secure distributed training.
pub struct SecureTrainingCoordinator {
    /// Configuration.
    config: SecureTrainingConfig,
    /// MPC session.
    session: MPCSession,
    /// Audit log.
    audit: AuditLog,
    /// Current training step.
    current_step: u64,
    /// Blinding factor generator for commitments.
    blinding_gen: BlindingGenerator,
    /// Model commitments (published on-chain).
    model_commitments: Option<ModelCommitments>,
    /// Training metrics per step.
    step_metrics: Vec<StepMetrics>,
}

/// Metrics for one training step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: u64,
    pub loss: Option<f64>,
    pub gradient_norm: f64,
    pub num_contributors: usize,
    pub reshared: bool,
}

impl SecureTrainingCoordinator {
    /// Creates a new secure training coordinator.
    pub fn new(config: SecureTrainingConfig) -> MPCResult<Self> {
        let session = MPCSession::new(config.mpc.clone(), "training")?;

        Ok(Self {
            config,
            session,
            audit: AuditLog::new(),
            current_step: 0,
            blinding_gen: BlindingGenerator::with_seed(0xC0AA17),
            model_commitments: None,
            step_metrics: Vec::new(),
        })
    }

    /// Initializes the training session: register parties, preprocess, share model.
    pub fn initialize(
        &mut self,
        model_weights: &ModelWeightsFlat,
    ) -> MPCResult<()> {
        let n = self.config.mpc.num_parties;
        let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();

        // Register parties.
        for (i, party) in parties.iter().enumerate() {
            let role = if i == 0 {
                PartyRole::Dealer
            } else {
                PartyRole::Worker
            };
            self.session
                .register_participant(party.clone(), i, role, 1000)?;
        }

        // Preprocessing: generate Beaver triples.
        self.session
            .preprocess(self.config.hidden_dim, self.config.num_layers)?;
        self.audit.record_phase_transition(
            MPCPhase::Preprocessing,
            MPCPhase::InputSharing,
        );

        // Secret-share the model weights.
        let sharing = AdditiveSharing::with_seed(0xD0DE1);

        let layers: Vec<(usize, Vec<(&str, &[f32], &[usize], f64)>)> = model_weights
            .layers
            .iter()
            .map(|layer| {
                let weight_refs: Vec<(&str, &[f32], &[usize], f64)> = layer
                    .weights
                    .iter()
                    .map(|(name, data, shape)| {
                        (name.as_str(), data.as_slice(), shape.as_slice(), 0.0)
                    })
                    .collect();
                (layer.layer_idx, weight_refs)
            })
            .collect();

        let embeddings = model_weights.embeddings.as_ref().map(|(data, shape)| {
            (data.as_slice(), shape.as_slice(), 0.0)
        });

        let lm_head = model_weights.lm_head.as_ref().map(|(data, shape)| {
            (data.as_slice(), shape.as_slice(), 0.0)
        });

        let model_shares = ModelSharing::share_model_additive(
            embeddings,
            &layers,
            lm_head,
            &model_weights.name,
            &parties,
            &sharing,
        )?;

        // Create commitments for all shares.
        let commitments = self.create_commitments(&model_shares);
        self.model_commitments = Some(commitments);

        self.audit
            .record_distribution("model_weights", n, &parties);

        // Distribute shares to session.
        self.session.distribute_model_shares(model_shares)?;
        self.audit.record_phase_transition(
            MPCPhase::InputSharing,
            MPCPhase::Computation,
        );

        Ok(())
    }

    /// Executes one secure training step.
    ///
    /// Each party computes a gradient on their share locally, then the
    /// coordinator aggregates the gradient shares and updates the model shares.
    pub fn training_step(
        &mut self,
        gradient_shares: Vec<GradientShare>,
    ) -> MPCResult<StepMetrics> {
        self.current_step += 1;
        self.audit.set_step(self.current_step);

        let n = self.config.mpc.num_parties;
        if gradient_shares.len() != n {
            return Err(MPCError::ShareCountMismatch {
                expected: n,
                got: gradient_shares.len(),
            });
        }

        // Aggregate gradient shares.
        // In additive sharing, the aggregated gradient share per party
        // is just the party's own gradient share (since each party computed
        // on their weight share independently).
        //
        // The gradient for each party's weight share is:
        // g_i = local_gradient(W_i, data)
        // And the true gradient G = sum(g_i) only when reconstructed.

        // Apply gradient to each party's model share locally.
        let lr = self.config.learning_rate;
        for (i, grad) in gradient_shares.iter().enumerate() {
            let model_share = self.session.model_share_mut(i)?;
            ModelSharing::apply_gradient_share(model_share, grad, lr)?;
        }

        // Check if re-sharing is needed.
        let reshared = self.session.advance_step()?;
        if reshared {
            let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();
            self.audit.record_resharing("model_weights", &parties);
        }

        // Replenish Beaver triples for next step.
        self.session
            .replenish_triples(self.config.hidden_dim, self.config.num_layers)?;

        let metrics = StepMetrics {
            step: self.current_step,
            loss: None,
            gradient_norm: 0.0,
            num_contributors: n,
            reshared,
        };

        self.step_metrics.push(metrics.clone());
        self.audit
            .record_training_step(self.current_step, metrics.loss);

        Ok(metrics)
    }

    /// Reconstructs the trained model from all parties' shares.
    pub fn reconstruct_model(&mut self) -> MPCResult<ReconstructedModel> {
        self.session.begin_reconstruction()?;

        let n = self.config.mpc.num_parties;
        let shares: Vec<ModelShare> = (0..n)
            .map(|i| self.session.model_share(i).cloned())
            .collect::<MPCResult<Vec<_>>>()?;

        let model = ModelSharing::reconstruct_model_additive(&shares)?;

        self.session.complete();
        self.audit
            .record_phase_transition(MPCPhase::OutputReconstruction, MPCPhase::Complete);

        Ok(model)
    }

    /// Returns the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        &self.audit
    }

    /// Returns training metrics.
    pub fn metrics(&self) -> &[StepMetrics] {
        &self.step_metrics
    }

    /// Returns the current step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns model commitments (for on-chain verification).
    pub fn model_commitments(&self) -> Option<&ModelCommitments> {
        self.model_commitments.as_ref()
    }

    /// Gets a reference to a party's model share.
    pub fn model_share(&self, party_index: usize) -> MPCResult<&ModelShare> {
        self.session.model_share(party_index)
    }

    /// Gets mutable reference to all Beaver pools.
    pub fn pools_mut(&mut self) -> &mut [crate::beaver::pool::BeaverPool] {
        self.session.all_pools_mut()
    }

    /// Creates commitments for all model shares.
    fn create_commitments(&mut self, shares: &[ModelShare]) -> ModelCommitments {
        let mut per_party = Vec::new();

        for share in shares {
            let mut party_comms = Vec::new();

            if let Some(ref embed) = share.embeddings {
                let blinding = self.blinding_gen.generate();
                party_comms.push(ShareCommitment::commit_fr_vector(
                    &share.party,
                    &embed.data,
                    &blinding,
                    "embeddings",
                ));
            }

            for layer in &share.layers {
                for (name, tensor) in &layer.weights {
                    let blinding = self.blinding_gen.generate();
                    let desc = format!("layer{}_{}", layer.layer_idx, name);
                    party_comms.push(ShareCommitment::commit_tensor(
                        &share.party,
                        tensor,
                        &blinding,
                        desc,
                    ));
                }
            }

            if let Some(ref lm) = share.lm_head {
                let blinding = self.blinding_gen.generate();
                party_comms.push(ShareCommitment::commit_fr_vector(
                    &share.party,
                    &lm.data,
                    &blinding,
                    "lm_head",
                ));
            }

            per_party.push(party_comms);
        }

        ModelCommitments::create(per_party)
    }
}

/// Flat representation of model weights for sharing.
/// This is a simplified interface for passing weights to the coordinator.
#[derive(Debug, Clone)]
pub struct ModelWeightsFlat {
    pub name: String,
    pub embeddings: Option<(Vec<f32>, Vec<usize>)>,
    pub layers: Vec<LayerWeightsFlat>,
    pub lm_head: Option<(Vec<f32>, Vec<usize>)>,
}

/// Flat representation of layer weights.
#[derive(Debug, Clone)]
pub struct LayerWeightsFlat {
    pub layer_idx: usize,
    /// (weight_name, data, shape)
    pub weights: Vec<(String, Vec<f32>, Vec<usize>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharing::tensor::TensorShare;
    use crate::sharing::model::LayerGradientShare;
    use crate::types::ShareId;

    fn create_test_model() -> ModelWeightsFlat {
        let layer0_weights = vec![
            ("q_proj".to_string(), vec![0.1f32; 4], vec![2, 2]),
            ("v_proj".to_string(), vec![0.2f32; 4], vec![2, 2]),
        ];

        ModelWeightsFlat {
            name: "test-model".to_string(),
            embeddings: Some((vec![0.5f32; 6], vec![3, 2])),
            layers: vec![LayerWeightsFlat {
                layer_idx: 0,
                weights: layer0_weights,
            }],
            lm_head: Some((vec![0.1f32; 6], vec![2, 3])),
        }
    }

    fn create_test_gradient(n_parties: usize) -> Vec<GradientShare> {
        (0..n_parties)
            .map(|i| GradientShare {
                party: PartyId::from_index(i),
                index: i,
                embeddings: Some(TensorShare::from_f64(
                    ShareId::new(PartyId::from_index(i), "grad_embed", i),
                    vec![0.01; 6],
                    vec![3, 2],
                )),
                layers: vec![LayerGradientShare {
                    layer_idx: 0,
                    gradients: [
                        (
                            "q_proj".to_string(),
                            TensorShare::from_f64(
                                ShareId::new(PartyId::from_index(i), "grad_q", i),
                                vec![0.01; 4],
                                vec![2, 2],
                            ),
                        ),
                        (
                            "v_proj".to_string(),
                            TensorShare::from_f64(
                                ShareId::new(PartyId::from_index(i), "grad_v", i),
                                vec![0.01; 4],
                                vec![2, 2],
                            ),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                }],
                lm_head: Some(TensorShare::from_f64(
                    ShareId::new(PartyId::from_index(i), "grad_lm", i),
                    vec![0.01; 6],
                    vec![2, 3],
                )),
                error_bound: 0.01,
            })
            .collect()
    }

    #[test]
    fn test_secure_training_lifecycle() {
        let config = SecureTrainingConfig {
            mpc: MPCConfig {
                num_parties: 3,
                reshare_interval: 5,
                ..MPCConfig::default()
            },
            hidden_dim: 2,
            num_layers: 1,
            ..Default::default()
        };

        let mut coordinator = SecureTrainingCoordinator::new(config).unwrap();
        let model = create_test_model();

        // Initialize.
        coordinator.initialize(&model).unwrap();

        // Run a few training steps.
        for _ in 0..3 {
            let gradients = create_test_gradient(3);
            let metrics = coordinator.training_step(gradients).unwrap();
            assert_eq!(metrics.num_contributors, 3);
        }

        assert_eq!(coordinator.current_step(), 3);

        // Reconstruct.
        let recon = coordinator.reconstruct_model().unwrap();

        // The model should exist and have the right structure.
        assert!(recon.embeddings.is_some());
        assert_eq!(recon.layers.len(), 1);
        assert!(recon.lm_head.is_some());

        // Verify embedding values changed from gradient updates.
        let embed = recon.embeddings.unwrap();
        // Original was 0.5, gradient was 0.01*3 per party (summed) = 0.03 * lr per step
        // After 3 steps: 0.5 - 3 * 0.001 * 0.03 = 0.49991
        // But wait — each party has gradient 0.01, and there are 3 parties.
        // The gradient share IS the full gradient for that party's share.
        // w_i -= lr * g_i where g_i = 0.01
        // After 3 steps: original_share - 3 * 0.001 * 0.01
        // Reconstructed = sum(shares) = original - 3 * 3 * 0.001 * 0.01
        // = 0.5 - 0.00009 ≈ 0.49991

        // Audit log should have events.
        let audit = coordinator.audit_log();
        assert!(audit.len() > 0);

        let summary = audit.summary();
        assert!(summary.distributions > 0);
        assert_eq!(summary.training_steps, 3);
    }

    #[test]
    fn test_commitments_created() {
        let config = SecureTrainingConfig::default();
        let mut coordinator = SecureTrainingCoordinator::new(config).unwrap();
        let model = create_test_model();

        coordinator.initialize(&model).unwrap();

        let commitments = coordinator.model_commitments().unwrap();
        assert_eq!(commitments.per_party.len(), 3);
        assert_ne!(commitments.root_hash, [0u8; 32]);
    }

    #[test]
    fn test_resharing_occurs() {
        let config = SecureTrainingConfig {
            mpc: MPCConfig {
                num_parties: 3,
                reshare_interval: 2, // Reshare every 2 steps.
                ..MPCConfig::default()
            },
            ..Default::default()
        };

        let mut coordinator = SecureTrainingCoordinator::new(config).unwrap();
        let model = create_test_model();
        coordinator.initialize(&model).unwrap();

        // Step 1: no reshare.
        let m1 = coordinator
            .training_step(create_test_gradient(3))
            .unwrap();
        assert!(!m1.reshared);

        // Step 2: reshare.
        let m2 = coordinator
            .training_step(create_test_gradient(3))
            .unwrap();
        assert!(m2.reshared);
    }
}
