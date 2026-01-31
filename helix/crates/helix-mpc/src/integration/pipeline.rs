//! End-to-end secure training pipeline.
//!
//! Provides a high-level API that orchestrates all MPC components
//! for a complete secure training run:
//! - Model sharing
//! - Beaver triple preprocessing
//! - Secure forward/backward passes
//! - Gradient aggregation on shares
//! - Model reconstruction

use crate::beaver::dealer::TrustedDealer;
use crate::beaver::pool::BeaverPool;
use crate::error::{MPCError, MPCResult};
use crate::integration::training::{SecureTrainingConfig, SecureTrainingCoordinator, ModelWeightsFlat};
use crate::protocols::activation::{ActivationType, SecureActivation};
use crate::protocols::arithmetic::SecureArithmetic;
use crate::protocols::matmul::SecureMatmul;
use crate::protocols::normalization::SecureNormalization;
use crate::security::audit::AuditLog;
use crate::sharing::model::{GradientShare, LayerGradientShare, ModelSharing, ReconstructedModel};
use crate::sharing::tensor::TensorShare;
use crate::sharing::{AdditiveSharing, SecretSharingScheme};
use crate::types::{MPCConfig, PartyId, ShareId};

/// High-level secure training pipeline.
///
/// Usage:
/// ```ignore
/// let pipeline = SecurePipeline::new(config);
/// pipeline.share_model(&weights);
///
/// for batch in data {
///     pipeline.training_step(&batch);
/// }
///
/// let model = pipeline.reconstruct();
/// ```
pub struct SecurePipeline {
    coordinator: SecureTrainingCoordinator,
    config: SecureTrainingConfig,
    initialized: bool,
}

impl SecurePipeline {
    /// Creates a new pipeline.
    pub fn new(config: SecureTrainingConfig) -> MPCResult<Self> {
        let coordinator = SecureTrainingCoordinator::new(config.clone())?;
        Ok(Self {
            coordinator,
            config,
            initialized: false,
        })
    }

    /// Creates a pipeline with default 3-party configuration.
    pub fn default_three_party(
        hidden_dim: usize,
        num_layers: usize,
        learning_rate: f64,
        total_steps: u64,
    ) -> MPCResult<Self> {
        let config = SecureTrainingConfig {
            mpc: MPCConfig::three_party(),
            learning_rate,
            total_steps,
            hidden_dim,
            num_layers,
            max_gradient_norm: 10.0,
        };
        Self::new(config)
    }

    /// Shares the model and initializes the MPC session.
    pub fn share_model(&mut self, weights: &ModelWeightsFlat) -> MPCResult<()> {
        self.coordinator.initialize(weights)?;
        self.initialized = true;
        Ok(())
    }

    /// Runs one training step with provided gradient shares.
    pub fn training_step(
        &mut self,
        gradient_shares: Vec<GradientShare>,
    ) -> MPCResult<StepResult> {
        if !self.initialized {
            return Err(MPCError::SessionError("Pipeline not initialized".into()));
        }

        let metrics = self.coordinator.training_step(gradient_shares)?;

        Ok(StepResult {
            step: metrics.step,
            loss: metrics.loss,
            reshared: metrics.reshared,
        })
    }

    /// Simulates a complete training step including gradient computation.
    ///
    /// This is a simplified version for the demo that:
    /// 1. Generates synthetic gradients for each party's share
    /// 2. Applies the gradients to model shares
    /// 3. Returns metrics
    ///
    /// In production, each party would compute gradients on their share
    /// using the secure NN layers.
    pub fn simulate_training_step(
        &mut self,
        gradient_scale: f64,
    ) -> MPCResult<StepResult> {
        if !self.initialized {
            return Err(MPCError::SessionError("Pipeline not initialized".into()));
        }

        let n = self.config.mpc.num_parties;

        // Generate synthetic gradient shares.
        // In the real system, each party computes on their weight share.
        let grad_shares = self.generate_synthetic_gradients(gradient_scale)?;

        self.training_step(grad_shares)
    }

    /// Reconstructs the trained model.
    pub fn reconstruct(&mut self) -> MPCResult<ReconstructedModel> {
        self.coordinator.reconstruct_model()
    }

    /// Returns the current step number.
    pub fn current_step(&self) -> u64 {
        self.coordinator.current_step()
    }

    /// Returns the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        self.coordinator.audit_log()
    }

    /// Returns a reference to the coordinator for advanced operations.
    pub fn coordinator(&self) -> &SecureTrainingCoordinator {
        &self.coordinator
    }

    /// Returns a mutable reference to the coordinator.
    pub fn coordinator_mut(&mut self) -> &mut SecureTrainingCoordinator {
        &mut self.coordinator
    }

    /// Generates synthetic gradient shares for testing/demo.
    fn generate_synthetic_gradients(
        &self,
        scale: f64,
    ) -> MPCResult<Vec<GradientShare>> {
        let n = self.config.mpc.num_parties;

        // Get the first party's model share to determine shapes.
        let template = self.coordinator.model_share(0)?;

        let mut grad_shares: Vec<GradientShare> = (0..n)
            .map(|i| GradientShare {
                party: PartyId::from_index(i),
                index: i,
                embeddings: None,
                layers: Vec::new(),
                lm_head: None,
                error_bound: 0.01,
            })
            .collect();

        // Generate gradient shares matching model structure.
        if let Some(ref embed) = template.embeddings {
            let sharing = AdditiveSharing::with_seed(
                0x6EAD + self.coordinator.current_step(),
            );
            let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();

            // Synthetic gradient: small random values scaled by `scale`.
            let grad_data: Vec<f64> = (0..embed.data.len())
                .map(|i| {
                    let base = ((i as f64 * 0.1).sin() * 0.01) * scale;
                    base
                })
                .collect();

            let shares = sharing
                .share_vector(&grad_data, "grad_embed", &parties)?;

            for (i, share) in shares.into_iter().enumerate() {
                grad_shares[i].embeddings = Some(TensorShare::new(
                    share.id,
                    share.values,
                    embed.shape.clone(),
                ));
            }
        }

        for layer in &template.layers {
            let mut layer_grads: Vec<LayerGradientShare> = (0..n)
                .map(|_| LayerGradientShare {
                    layer_idx: layer.layer_idx,
                    gradients: std::collections::HashMap::new(),
                })
                .collect();

            for (name, tensor) in &layer.weights {
                let sharing = AdditiveSharing::with_seed(
                    0x6EAD + self.coordinator.current_step() + layer.layer_idx as u64,
                );
                let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();

                let grad_data: Vec<f64> = (0..tensor.data.len())
                    .map(|i| ((i as f64 * 0.2).cos() * 0.01) * scale)
                    .collect();

                let shares = sharing
                    .share_vector(&grad_data, &format!("grad_{}", name), &parties)?;

                for (i, share) in shares.into_iter().enumerate() {
                    layer_grads[i].gradients.insert(
                        name.clone(),
                        TensorShare::new(share.id, share.values, tensor.shape.clone()),
                    );
                }
            }

            for (i, lg) in layer_grads.into_iter().enumerate() {
                grad_shares[i].layers.push(lg);
            }
        }

        if let Some(ref lm) = template.lm_head {
            let sharing = AdditiveSharing::with_seed(
                0x6EAD + self.coordinator.current_step() + 999,
            );
            let parties: Vec<PartyId> = (0..n).map(PartyId::from_index).collect();

            let grad_data: Vec<f64> = (0..lm.data.len())
                .map(|i| ((i as f64 * 0.3).sin() * 0.01) * scale)
                .collect();

            let shares = sharing
                .share_vector(&grad_data, "grad_lm", &parties)?;

            for (i, share) in shares.into_iter().enumerate() {
                grad_shares[i].lm_head = Some(TensorShare::new(
                    share.id,
                    share.values,
                    lm.shape.clone(),
                ));
            }
        }

        Ok(grad_shares)
    }
}

/// Result of a training step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step: u64,
    pub loss: Option<f64>,
    pub reshared: bool,
}

/// Runs a complete demo training session.
///
/// This is a convenience function for the ETHDenver demo that:
/// 1. Creates a small model
/// 2. Secret-shares it among 3 parties
/// 3. Runs N training steps with synthetic gradients
/// 4. Reconstructs the model
/// 5. Returns the final model and audit log
pub fn run_demo_training(
    num_steps: u64,
) -> MPCResult<DemoResult> {
    let hidden_dim = 8;
    let num_layers = 2;

    // Create a small model.
    let model = ModelWeightsFlat {
        name: "demo-transformer".to_string(),
        embeddings: Some((vec![0.1f32; hidden_dim * 16], vec![16, hidden_dim])),
        layers: (0..num_layers)
            .map(|idx| crate::integration::training::LayerWeightsFlat {
                layer_idx: idx,
                weights: vec![
                    (
                        "q_proj".to_string(),
                        vec![0.02f32; hidden_dim * hidden_dim],
                        vec![hidden_dim, hidden_dim],
                    ),
                    (
                        "k_proj".to_string(),
                        vec![0.02f32; hidden_dim * hidden_dim],
                        vec![hidden_dim, hidden_dim],
                    ),
                    (
                        "v_proj".to_string(),
                        vec![0.02f32; hidden_dim * hidden_dim],
                        vec![hidden_dim, hidden_dim],
                    ),
                    (
                        "o_proj".to_string(),
                        vec![0.02f32; hidden_dim * hidden_dim],
                        vec![hidden_dim, hidden_dim],
                    ),
                    (
                        "mlp_up".to_string(),
                        vec![0.02f32; hidden_dim * hidden_dim * 4],
                        vec![hidden_dim, hidden_dim * 4],
                    ),
                    (
                        "mlp_down".to_string(),
                        vec![0.02f32; hidden_dim * 4 * hidden_dim],
                        vec![hidden_dim * 4, hidden_dim],
                    ),
                ],
            })
            .collect(),
        lm_head: Some((vec![0.1f32; hidden_dim * 16], vec![hidden_dim, 16])),
    };

    let mut pipeline = SecurePipeline::default_three_party(
        hidden_dim,
        num_layers,
        0.001,
        num_steps,
    )?;

    pipeline.share_model(&model)?;

    let mut step_results = Vec::new();
    for _ in 0..num_steps {
        let result = pipeline.simulate_training_step(1.0)?;
        step_results.push(result);
    }

    let reconstructed = pipeline.reconstruct()?;
    let audit_summary = pipeline.audit_log().summary();

    Ok(DemoResult {
        model: reconstructed,
        steps: step_results,
        audit_summary_text: format!("{}", audit_summary),
        total_events: audit_summary.total_events,
    })
}

/// Result of a demo training run.
pub struct DemoResult {
    pub model: ReconstructedModel,
    pub steps: Vec<StepResult>,
    pub audit_summary_text: String,
    pub total_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_training() {
        let result = run_demo_training(5).unwrap();

        assert_eq!(result.steps.len(), 5);
        assert!(result.total_events > 0);
        assert!(result.model.embeddings.is_some());
        assert_eq!(result.model.layers.len(), 2);
        assert!(result.model.lm_head.is_some());
    }

    #[test]
    fn test_pipeline_lifecycle() {
        let config = SecureTrainingConfig {
            mpc: MPCConfig::three_party(),
            hidden_dim: 4,
            num_layers: 1,
            learning_rate: 0.01,
            total_steps: 10,
            max_gradient_norm: 10.0,
        };

        let mut pipeline = SecurePipeline::new(config).unwrap();

        let model = ModelWeightsFlat {
            name: "test".to_string(),
            embeddings: Some((vec![0.5f32; 8], vec![4, 2])),
            layers: vec![crate::integration::training::LayerWeightsFlat {
                layer_idx: 0,
                weights: vec![(
                    "w".to_string(),
                    vec![0.1f32; 4],
                    vec![2, 2],
                )],
            }],
            lm_head: None,
        };

        pipeline.share_model(&model).unwrap();

        for _ in 0..3 {
            let result = pipeline.simulate_training_step(0.5).unwrap();
            assert!(result.step > 0);
        }

        let recon = pipeline.reconstruct().unwrap();
        assert!(recon.embeddings.is_some());
    }
}
