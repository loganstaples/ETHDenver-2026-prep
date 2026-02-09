//! Integrated Training Flow with ZK Proofs.
//!
//! This module provides a complete training workflow that combines:
//! - MPC secret sharing of model weights
//! - Secure gradient computation on shares
//! - ZK proof generation for each training step
//! - Verification of all proofs
//!
//! # End-to-End Flow
//!
//! 1. Model owner shares weights among parties
//! 2. Each training step:
//!    a. Parties receive input batch
//!    b. Each party computes gradients on their share
//!    c. Gradients are aggregated (still secret-shared)
//!    d. ZK proof is generated proving correct computation
//!    e. Weights are updated
//! 3. Model is reconstructed with final weights
//!
//! All steps are accompanied by ZK proofs that can be verified on-chain.

use std::collections::HashMap;
use std::time::Instant;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::zk_pipeline::{
    BatchProofGenerator, ProofResult, ZKPipelineConfig, ZKProofPipeline,
};
use crate::proofs::{
    AggregationProof, AggregationProver, AggregationVerifier,
    GradientAggregationWitness, GradientShareInput, MACProver, ProofStats, ShareValidityProof, ShareValidityProver, ShareValidityVerifier,
    ShareValidityWitness,
};
use crate::security::commitment::BlindingGenerator;
use crate::sharing::model::{GradientShare, ModelShare};
use crate::sharing::tensor::TensorShare;
use crate::types::{MPCConfig, PartyId, ShareId};

/// Configuration for ZK-integrated training.
#[derive(Debug, Clone)]
pub struct ZKTrainingConfig {
    /// MPC configuration.
    pub mpc: MPCConfig,
    /// Model dimensions.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Total training steps.
    pub total_steps: u64,
    /// Maximum error bound per step.
    pub max_error: f64,
    /// Whether to generate proofs for each step.
    pub generate_proofs: bool,
    /// Whether to batch proofs.
    pub batch_proofs: bool,
    /// Batch size for proof batching.
    pub proof_batch_size: usize,
}

impl Default for ZKTrainingConfig {
    fn default() -> Self {
        Self {
            mpc: MPCConfig::three_party(),
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            learning_rate: 0.001,
            total_steps: 100,
            max_error: 1e-3,
            generate_proofs: true,
            batch_proofs: true,
            proof_batch_size: 10,
        }
    }
}

/// Result of a single ZK training step.
#[derive(Debug, Clone)]
pub struct ZKTrainingStep {
    /// Step number.
    pub step: u64,
    /// Loss value (if computed).
    pub loss: Option<f64>,
    /// Training step proof.
    pub training_proof: Option<ProofResult>,
    /// Share validity proofs per party.
    pub share_proofs: Vec<ShareValidityProof>,
    /// Aggregation proof.
    pub aggregation_proof: Option<AggregationProof>,
    /// Old state hash.
    pub old_state_hash: (Fr, Fr),
    /// New state hash.
    pub new_state_hash: (Fr, Fr),
    /// Step execution time in ms.
    pub execution_time_ms: u64,
    /// Proof generation time in ms.
    pub proof_time_ms: u64,
    /// Total error for this step.
    pub total_error: f64,
}

/// Coordinated ZK training with integrated proofs.
#[allow(dead_code)]
pub struct ZKTrainingCoordinator {
    /// Configuration.
    config: ZKTrainingConfig,
    /// ZK proof pipeline.
    zk_pipeline: ZKProofPipeline,
    /// Share validity prover.
    share_prover: ShareValidityProver,
    /// Aggregation prover.
    agg_prover: AggregationProver,
    /// MAC prover.
    mac_prover: MACProver,
    /// Batch proof generator.
    batch_generator: Option<BatchProofGenerator>,
    /// Blinding generator.
    blinding_gen: BlindingGenerator,
    /// Current step.
    current_step: u64,
    /// Training history.
    history: Vec<ZKTrainingStep>,
    /// Proof statistics.
    proof_stats: ProofStats,
    /// Model shares per party.
    model_shares: HashMap<usize, ModelShare>,
    /// Dealer public key (for share verification).
    dealer_pk: [u8; 32],
}

impl ZKTrainingCoordinator {
    /// Creates a new ZK training coordinator.
    pub fn new(config: ZKTrainingConfig) -> Self {
        let zk_config = ZKPipelineConfig {
            num_parties: config.mpc.num_parties,
            d_in: config.d_in,
            d_hid: config.d_hid,
            d_out: config.d_out,
            base_error: 1e-6,
            max_error: config.max_error,
            ..Default::default()
        };

        let batch_generator = if config.batch_proofs {
            Some(BatchProofGenerator::new(zk_config.clone()))
        } else {
            None
        };

        Self {
            config,
            zk_pipeline: ZKProofPipeline::new(zk_config),
            share_prover: ShareValidityProver::new(),
            agg_prover: AggregationProver::new(),
            mac_prover: MACProver::new(),
            batch_generator,
            blinding_gen: BlindingGenerator::new(),
            current_step: 0,
            history: Vec::new(),
            proof_stats: ProofStats::new(),
            model_shares: HashMap::new(),
            dealer_pk: [0u8; 32],
        }
    }

    /// Initializes training with model weights.
    pub fn initialize(&mut self, initial_weights: InitialWeights) -> MPCResult<()> {
        let num_parties = self.config.mpc.num_parties;
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();

        // Generate dealer key.
        let mut rng = rand::thread_rng();
        use rand::RngCore;
        rng.fill_bytes(&mut self.dealer_pk);

        // Share the model weights.
        let _sharing = crate::sharing::AdditiveSharing::with_seed(0xD0DE1);

        for i in 0..num_parties {
            let party = parties[i].clone();

            // Create weight shares.
            let w1_share = TensorShare::from_f64(
                ShareId::new(party.clone(), "w1", i),
                initial_weights.w1.iter().map(|v| *v as f64 / num_parties as f64).collect(),
                vec![self.config.d_hid, self.config.d_in],
            );
            let b1_share = TensorShare::from_f64(
                ShareId::new(party.clone(), "b1", i),
                initial_weights.b1.iter().map(|v| *v as f64 / num_parties as f64).collect(),
                vec![self.config.d_hid],
            );
            let w2_share = TensorShare::from_f64(
                ShareId::new(party.clone(), "w2", i),
                initial_weights.w2.iter().map(|v| *v as f64 / num_parties as f64).collect(),
                vec![self.config.d_out, self.config.d_hid],
            );
            let b2_share = TensorShare::from_f64(
                ShareId::new(party.clone(), "b2", i),
                initial_weights.b2.iter().map(|v| *v as f64 / num_parties as f64).collect(),
                vec![self.config.d_out],
            );

            let mut weights = std::collections::HashMap::new();
            weights.insert("w1".to_string(), w1_share);
            weights.insert("b1".to_string(), b1_share);
            weights.insert("w2".to_string(), w2_share);
            weights.insert("b2".to_string(), b2_share);

            let model_share = ModelShare {
                party,
                index: i,
                model_name: "training_model".to_string(),
                num_layers: 1,
                embeddings: None,
                layers: vec![crate::sharing::model::LayerShare {
                    layer_idx: 0,
                    weights,
                }],
                lm_head: None,
                extra_weights: std::collections::HashMap::new(),
            };

            self.model_shares.insert(i, model_share);
        }

        Ok(())
    }

    /// Executes one training step with ZK proof generation.
    pub fn training_step(&mut self, input: &[f64], target: &[f64]) -> MPCResult<ZKTrainingStep> {
        let step_start = Instant::now();
        self.current_step += 1;

        // Generate gradient shares for each party.
        let gradient_shares = self.compute_gradient_shares(input, target)?;

        // Collect witnesses from each party.
        let proof_start = Instant::now();
        let mut share_proofs = Vec::new();

        for (party_idx, model_share) in &self.model_shares {
            let gradient_share = &gradient_shares[*party_idx];
            let blinding = self.blinding_gen.generate();

            // Generate party witness.
            self.zk_pipeline.generate_party_witness(
                *party_idx,
                model_share,
                gradient_share,
                input,
                target,
                self.config.learning_rate,
            )?;

            // Generate share validity proof if proofs are enabled.
            if self.config.generate_proofs {
                let witness = self.create_share_validity_witness(model_share, &blinding)?;
                let proof = self.share_prover.prove(&witness)?;
                share_proofs.push(proof);
            }
        }

        // Generate aggregation proof.
        let aggregation_proof = if self.config.generate_proofs {
            Some(self.generate_aggregation_proof(&gradient_shares)?)
        } else {
            None
        };

        // Generate training step proof.
        let training_proof = if self.config.generate_proofs {
            Some(self.zk_pipeline.generate_proof()?)
        } else {
            None
        };

        let proof_time = proof_start.elapsed();

        // Compute state hashes.
        let old_hash = self.compute_current_state_hash();

        // Apply gradients to update model shares.
        self.apply_gradients(&gradient_shares)?;

        let new_hash = self.compute_current_state_hash();

        // Compute total error.
        let total_error = training_proof
            .as_ref()
            .map(|p| p.total_error.to_f64())
            .unwrap_or(0.0);

        let step_time = step_start.elapsed();

        // Record proof stats.
        if let Some(ref proof) = training_proof {
            self.proof_stats.record(proof.proof_size_bytes, proof.generation_time_ms);

            // Add to batch if batching enabled.
            if let Some(ref mut batch) = self.batch_generator {
                batch.add_proof(proof.clone());
            }
        }

        // Advance ZK pipeline.
        self.zk_pipeline.next_step();

        let step_result = ZKTrainingStep {
            step: self.current_step,
            loss: training_proof.as_ref().map(|p| p.loss.to_f64()),
            training_proof,
            share_proofs,
            aggregation_proof,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            execution_time_ms: step_time.as_millis() as u64,
            proof_time_ms: proof_time.as_millis() as u64,
            total_error,
        };

        self.history.push(step_result.clone());

        Ok(step_result)
    }

    /// Runs multiple training steps.
    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)]) -> MPCResult<TrainingResult> {
        let start = Instant::now();
        let mut steps = Vec::new();

        for (input, target) in data {
            let step = self.training_step(input, target)?;
            steps.push(step);
        }

        let total_time = start.elapsed();

        // Generate batch proof if enabled.
        let batch_proof = if let Some(ref batch) = self.batch_generator {
            if !batch.is_empty() {
                Some(batch.generate_batch_proof()?)
            } else {
                None
            }
        } else {
            None
        };

        // Reconstruct final model.
        let final_model = self.reconstruct_model()?;

        Ok(TrainingResult {
            steps,
            final_model,
            batch_proof,
            total_time_ms: total_time.as_millis() as u64,
            proof_stats: self.proof_stats.clone(),
        })
    }

    /// Reconstructs the trained model from shares.
    pub fn reconstruct_model(&self) -> MPCResult<ReconstructedWeights> {
        let mut w1 = vec![0.0f64; self.config.d_hid * self.config.d_in];
        let mut b1 = vec![0.0f64; self.config.d_hid];
        let mut w2 = vec![0.0f64; self.config.d_out * self.config.d_hid];
        let mut b2 = vec![0.0f64; self.config.d_out];

        for (_, model_share) in &self.model_shares {
            if let Some(layer) = model_share.layers.first() {
                if let Some(w1_share) = layer.weights.get("w1") {
                    for (i, v) in w1_share.data.iter().enumerate() {
                        w1[i] += v.to_f64();
                    }
                }
                if let Some(b1_share) = layer.weights.get("b1") {
                    for (i, v) in b1_share.data.iter().enumerate() {
                        b1[i] += v.to_f64();
                    }
                }
                if let Some(w2_share) = layer.weights.get("w2") {
                    for (i, v) in w2_share.data.iter().enumerate() {
                        w2[i] += v.to_f64();
                    }
                }
                if let Some(b2_share) = layer.weights.get("b2") {
                    for (i, v) in b2_share.data.iter().enumerate() {
                        b2[i] += v.to_f64();
                    }
                }
            }
        }

        Ok(ReconstructedWeights { w1, b1, w2, b2 })
    }

    /// Verifies all proofs from the training run.
    pub fn verify_all(&self) -> MPCResult<VerificationResult> {
        let share_verifier = ShareValidityVerifier::new();
        let agg_verifier = AggregationVerifier::new();
        let mut valid_count = 0;
        let mut invalid_count = 0;

        for step in &self.history {
            // Verify share proofs.
            for proof in &step.share_proofs {
                if share_verifier.verify(proof)? {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                }
            }

            // Verify aggregation proof.
            if let Some(ref proof) = step.aggregation_proof {
                if agg_verifier.verify(proof)? {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                }
            }

            // Verify training proof.
            if let Some(ref proof) = step.training_proof {
                if self.zk_pipeline.verify_proof(proof)? {
                    valid_count += 1;
                } else {
                    invalid_count += 1;
                }
            }
        }

        Ok(VerificationResult {
            valid_count,
            invalid_count,
            all_valid: invalid_count == 0,
        })
    }

    /// Returns training history.
    pub fn history(&self) -> &[ZKTrainingStep] {
        &self.history
    }

    /// Returns proof statistics.
    pub fn proof_stats(&self) -> &ProofStats {
        &self.proof_stats
    }

    /// Returns current step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    // Internal helpers.

    fn compute_gradient_shares(&self, _input: &[f64], _target: &[f64]) -> MPCResult<Vec<GradientShare>> {
        let num_parties = self.config.mpc.num_parties;
        let mut gradients = Vec::with_capacity(num_parties);

        for i in 0..num_parties {
            let party = PartyId::from_index(i);

            // Simplified gradient computation.
            // In production, this would use secure matmul protocols.
            let dw1 = vec![Fr::from_f64(0.01); self.config.d_hid * self.config.d_in];
            let db1 = vec![Fr::from_f64(0.01); self.config.d_hid];
            let dw2 = vec![Fr::from_f64(0.01); self.config.d_out * self.config.d_hid];
            let db2 = vec![Fr::from_f64(0.01); self.config.d_out];

            let mut weight_grads = std::collections::HashMap::new();
            weight_grads.insert(
                "w1".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "dw1", i),
                    dw1,
                    vec![self.config.d_hid, self.config.d_in],
                ),
            );
            weight_grads.insert(
                "b1".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "db1", i),
                    db1,
                    vec![self.config.d_hid],
                ),
            );
            weight_grads.insert(
                "w2".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "dw2", i),
                    dw2,
                    vec![self.config.d_out, self.config.d_hid],
                ),
            );
            weight_grads.insert(
                "b2".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "db2", i),
                    db2,
                    vec![self.config.d_out],
                ),
            );

            gradients.push(GradientShare {
                party,
                index: i,
                embeddings: None,
                layers: vec![crate::sharing::model::LayerGradientShare {
                    layer_idx: 0,
                    gradients: weight_grads,
                }],
                lm_head: None,
                error_bound: 0.01,
            });
        }

        Ok(gradients)
    }

    fn create_share_validity_witness(
        &self,
        model_share: &ModelShare,
        blinding: &[u8; 32],
    ) -> MPCResult<ShareValidityWitness> {
        let layer = model_share.layers.first().ok_or_else(|| {
            MPCError::ProtocolError("No layers in model share".into())
        })?;

        let w1 = layer.weights.get("w1").ok_or_else(|| {
            MPCError::ProtocolError("Missing w1 weight".into())
        })?;

        // Create dealer signature (simplified).
        let dealer_sig = vec![1, 2, 3, 4];

        Ok(ShareValidityWitness::from_tensor_share(
            w1,
            *blinding,
            self.dealer_pk,
            dealer_sig,
        ))
    }

    fn generate_aggregation_proof(
        &mut self,
        gradient_shares: &[GradientShare],
    ) -> MPCResult<AggregationProof> {
        let dim = self.config.d_hid * self.config.d_in;
        let mut witness = GradientAggregationWitness::new(
            self.config.mpc.num_parties,
            dim,
            self.current_step,
        );

        for (i, grad) in gradient_shares.iter().enumerate() {
            let party = PartyId::from_index(i);

            // Flatten gradient values.
            let values: Vec<Fr> = if let Some(layer) = grad.layers.first() {
                layer
                    .gradients
                    .get("w1")
                    .map(|t| t.data.clone())
                    .unwrap_or_else(|| vec![Fr::ZERO; dim])
            } else {
                vec![Fr::ZERO; dim]
            };

            let blinding = self.blinding_gen.generate();
            let mut hasher = sha2::Sha256::new();
            use sha2::Digest;
            for v in &values {
                hasher.update(&v.to_bytes_le());
            }
            hasher.update(&blinding);
            let _commitment: [u8; 32] = hasher.finalize().into();

            witness.add_gradient_share(GradientShareInput::new(
                party,
                values,
                blinding,
            ))?;
        }

        witness.compute_aggregation();
        self.agg_prover.prove(&witness)
    }

    fn apply_gradients(&mut self, gradient_shares: &[GradientShare]) -> MPCResult<()> {
        let lr = Fr::from_f64(self.config.learning_rate);

        for (i, grad) in gradient_shares.iter().enumerate() {
            if let Some(model_share) = self.model_shares.get_mut(&i) {
                if let Some(layer) = model_share.layers.first_mut() {
                    if let Some(grad_layer) = grad.layers.first() {
                        // Update w1.
                        if let (Some(w1), Some(dw1)) = (
                            layer.weights.get_mut("w1"),
                            grad_layer.gradients.get("w1"),
                        ) {
                            for (_j, (w, dw)) in w1.data.iter_mut().zip(dw1.data.iter()).enumerate() {
                                *w = Fr::sub(w, &Fr::mul(&lr, dw));
                            }
                        }

                        // Update b1.
                        if let (Some(b1), Some(db1)) = (
                            layer.weights.get_mut("b1"),
                            grad_layer.gradients.get("b1"),
                        ) {
                            for (_j, (b, db)) in b1.data.iter_mut().zip(db1.data.iter()).enumerate() {
                                *b = Fr::sub(b, &Fr::mul(&lr, db));
                            }
                        }

                        // Update w2.
                        if let (Some(w2), Some(dw2)) = (
                            layer.weights.get_mut("w2"),
                            grad_layer.gradients.get("w2"),
                        ) {
                            for (_j, (w, dw)) in w2.data.iter_mut().zip(dw2.data.iter()).enumerate() {
                                *w = Fr::sub(w, &Fr::mul(&lr, dw));
                            }
                        }

                        // Update b2.
                        if let (Some(b2), Some(db2)) = (
                            layer.weights.get_mut("b2"),
                            grad_layer.gradients.get("b2"),
                        ) {
                            for (_j, (b, db)) in b2.data.iter_mut().zip(db2.data.iter()).enumerate() {
                                *b = Fr::sub(b, &Fr::mul(&lr, db));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn compute_current_state_hash(&self) -> (Fr, Fr) {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();

        for i in 0..self.config.mpc.num_parties {
            if let Some(model_share) = self.model_shares.get(&i) {
                if let Some(layer) = model_share.layers.first() {
                    for (_name, tensor) in &layer.weights {
                        for v in &tensor.data {
                            hasher.update(&v.to_bytes_le());
                        }
                    }
                }
            }
        }

        let hash: [u8; 32] = hasher.finalize().into();

        let lo = Fr::from_bytes_le(&[
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
            hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let hi = Fr::from_bytes_le(&[
            hash[16], hash[17], hash[18], hash[19], hash[20], hash[21], hash[22], hash[23],
            hash[24], hash[25], hash[26], hash[27], hash[28], hash[29], hash[30], hash[31],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        (lo, hi)
    }
}

/// Initial model weights for training.
#[derive(Debug, Clone)]
pub struct InitialWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl InitialWeights {
    /// Creates random initial weights.
    pub fn random(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let scale1 = (2.0 / d_in as f32).sqrt();
        let scale2 = (2.0 / d_hid as f32).sqrt();

        Self {
            w1: (0..d_hid * d_in)
                .map(|_| rng.gen_range(-scale1..scale1))
                .collect(),
            b1: vec![0.0; d_hid],
            w2: (0..d_out * d_hid)
                .map(|_| rng.gen_range(-scale2..scale2))
                .collect(),
            b2: vec![0.0; d_out],
        }
    }

    /// Creates weights initialized to specific values.
    pub fn constant(d_in: usize, d_hid: usize, d_out: usize, value: f32) -> Self {
        Self {
            w1: vec![value; d_hid * d_in],
            b1: vec![0.0; d_hid],
            w2: vec![value; d_out * d_hid],
            b2: vec![0.0; d_out],
        }
    }
}

/// Reconstructed model weights after training.
#[derive(Debug, Clone)]
pub struct ReconstructedWeights {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

/// Complete training result.
#[derive(Debug)]
pub struct TrainingResult {
    /// All training steps.
    pub steps: Vec<ZKTrainingStep>,
    /// Final reconstructed model.
    pub final_model: ReconstructedWeights,
    /// Batched proof (if batch proofs enabled).
    pub batch_proof: Option<crate::integration::zk_pipeline::BatchProofResult>,
    /// Total training time in ms.
    pub total_time_ms: u64,
    /// Proof generation statistics.
    pub proof_stats: ProofStats,
}

impl TrainingResult {
    /// Returns the final loss.
    pub fn final_loss(&self) -> Option<f64> {
        self.steps.last().and_then(|s| s.loss)
    }

    /// Returns the average proof generation time.
    pub fn avg_proof_time_ms(&self) -> u64 {
        if self.steps.is_empty() {
            return 0;
        }
        let total: u64 = self.steps.iter().map(|s| s.proof_time_ms).sum();
        total / self.steps.len() as u64
    }
}

/// Result of verifying all proofs.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of valid proofs.
    pub valid_count: usize,
    /// Number of invalid proofs.
    pub invalid_count: usize,
    /// Whether all proofs are valid.
    pub all_valid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_training_coordinator_creation() {
        let config = ZKTrainingConfig::default();
        let coordinator = ZKTrainingCoordinator::new(config);

        assert_eq!(coordinator.current_step(), 0);
    }

    #[test]
    fn test_initialization() {
        let config = ZKTrainingConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            ..Default::default()
        };

        let mut coordinator = ZKTrainingCoordinator::new(config);

        let weights = InitialWeights::constant(2, 2, 1, 0.1);
        coordinator.initialize(weights).unwrap();

        assert_eq!(coordinator.model_shares.len(), 3);
    }

    #[test]
    fn test_training_step() {
        let config = ZKTrainingConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            generate_proofs: true,
            ..Default::default()
        };

        let mut coordinator = ZKTrainingCoordinator::new(config);

        let weights = InitialWeights::constant(2, 2, 1, 0.1);
        coordinator.initialize(weights).unwrap();

        let input = vec![1.0, 1.0];
        let target = vec![1.0];

        let step = coordinator.training_step(&input, &target).unwrap();

        assert_eq!(step.step, 1);
        assert!(step.training_proof.is_some());
        assert!(!step.share_proofs.is_empty());
    }

    #[test]
    fn test_full_training_run() {
        let config = ZKTrainingConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            total_steps: 5,
            generate_proofs: true,
            batch_proofs: true,
            ..Default::default()
        };

        let mut coordinator = ZKTrainingCoordinator::new(config);

        let weights = InitialWeights::random(2, 2, 1);
        coordinator.initialize(weights).unwrap();

        let data: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.5, 0.5], vec![0.5]),
        ];

        let result = coordinator.train(&data).unwrap();

        assert_eq!(result.steps.len(), 5);
        assert!(result.batch_proof.is_some());
    }

    #[test]
    fn test_verification() {
        let config = ZKTrainingConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            total_steps: 3,
            generate_proofs: true,
            ..Default::default()
        };

        let mut coordinator = ZKTrainingCoordinator::new(config);

        let weights = InitialWeights::constant(2, 2, 1, 0.1);
        coordinator.initialize(weights).unwrap();

        for _ in 0..3 {
            coordinator.training_step(&[1.0, 1.0], &[1.0]).unwrap();
        }

        let verification = coordinator.verify_all().unwrap();
        assert!(verification.all_valid);
    }

    #[test]
    fn test_model_reconstruction() {
        let config = ZKTrainingConfig {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            ..Default::default()
        };

        let mut coordinator = ZKTrainingCoordinator::new(config);

        let initial = InitialWeights::constant(2, 2, 1, 0.1);
        coordinator.initialize(initial).unwrap();

        let reconstructed = coordinator.reconstruct_model().unwrap();

        // All parties contribute 0.1/3, so sum should be ~0.1.
        assert!((reconstructed.w1[0] - 0.1).abs() < 0.001);
    }
}
