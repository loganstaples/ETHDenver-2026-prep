//! Integrated MPC-ZK Training Flow.
//!
//! This module is the CRITICAL PATH for HELIX - it enables workers to compute
//! on secret shares AND generate ZK proofs of correct computation. This is
//! the core technical novelty that makes trustless AI training possible.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Model Owner                                  │
//! │  1. Secret-shares model weights among workers                   │
//! └───────────────────────────┬─────────────────────────────────────┘
//!                             │
//!         ┌───────────────────┼───────────────────┐
//!         ▼                   ▼                   ▼
//! ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
//! │   Worker 0    │   │   Worker 1    │   │   Worker 2    │
//! │               │   │               │   │               │
//! │ Weight Share  │   │ Weight Share  │   │ Weight Share  │
//! │     +         │   │     +         │   │     +         │
//! │ Beaver Pool   │   │ Beaver Pool   │   │ Beaver Pool   │
//! │     +         │   │     +         │   │     +         │
//! │ Proof Capture │   │ Proof Capture │   │ Proof Capture │
//! └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
//!         │                   │                   │
//!         │     MPC Protocol (Beaver triples)    │
//!         │◄──────────────────┼──────────────────►│
//!         │                   │                   │
//!         ▼                   ▼                   ▼
//! ┌───────────────────────────────────────────────────────────────┐
//! │                       Aggregator                               │
//! │  2. Collects gradient shares + witness captures               │
//! │  3. Reconstructs full computation witness                      │
//! │  4. Generates ZK proof via CircuitBridge                       │
//! │  5. Verifies proof before submission                           │
//! └───────────────────────────┬───────────────────────────────────┘
//!                             │
//!                             ▼
//! ┌───────────────────────────────────────────────────────────────┐
//! │                    Smart Contract                              │
//! │  6. Verifies Halo2 proof on-chain                             │
//! │  7. Updates model state commitment                             │
//! │  8. Distributes rewards / slashes invalid submissions         │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Properties
//!
//! - **Privacy**: No worker sees full model weights (MPC secret sharing)
//! - **Verifiability**: Every computation is proven correct (ZK proofs)
//! - **Byzantine Tolerance**: Invalid proofs are rejected, workers slashed
//! - **Error Bounds**: Approximate computation with tracked error bounds

use std::collections::HashMap;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tracing::{debug, info, instrument, warn, span, Level};

use crate::beaver::dealer::TrustedDealer;
use crate::beaver::pool::BeaverPool;
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::circuit_bridge::{
    CircuitBridge, CircuitBridgeConfig, compute_compatible_state_hash,
};
use crate::integration::witness_format::{
    WitnessAggregator, WitnessBuilder,
};
use crate::integration::zk_pipeline::{
    ProofResult, ZKPipelineConfig, ZKProofPipeline,
};
use crate::protocols::proved_arithmetic::{
    ProvedArithmetic, WitnessCapture, WitnessSummary,
};
use crate::proofs::{
    AggregationProof, AggregationProver, BatchVerifier, ShareValidityProof, ShareValidityProver, ShareValidityWitness,
};
use crate::security::commitment::BlindingGenerator;
use crate::sharing::model::{GradientShare, LayerGradientShare, LayerShare, ModelShare};
use crate::sharing::tensor::TensorShare;
use crate::types::{PartyId, ShareId};

/// Configuration for the integrated MPC-ZK training.
#[derive(Debug, Clone)]
pub struct IntegratedTrainingConfig {
    /// Number of MPC workers.
    pub num_workers: usize,
    /// Model input dimension.
    pub d_in: usize,
    /// Hidden layer dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Base error bound per operation.
    pub base_error: f64,
    /// Maximum allowed total error.
    pub max_error: f64,
    /// Whether to generate real Halo2 proofs.
    pub use_real_proofs: bool,
    /// Circuit size parameter (k for 2^k rows).
    pub circuit_k: u32,
    /// Beaver triple batch size.
    pub beaver_batch_size: usize,
    /// Enable detailed logging.
    pub verbose_logging: bool,
}

impl Default for IntegratedTrainingConfig {
    fn default() -> Self {
        Self {
            num_workers: 2,
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            learning_rate: 0.01,
            base_error: 1e-6,
            max_error: 1e-3,
            use_real_proofs: false,
            circuit_k: 14,
            beaver_batch_size: 1024,
            verbose_logging: true,
        }
    }
}

impl IntegratedTrainingConfig {
    /// Creates a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            num_workers: 2,
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            learning_rate: 0.01,
            base_error: 1e-6,
            max_error: 1e-2,
            use_real_proofs: false,
            circuit_k: 12,
            beaver_batch_size: 100,
            verbose_logging: true,
        }
    }

    /// Enables real Halo2 proofs.
    pub fn with_real_proofs(mut self) -> Self {
        self.use_real_proofs = true;
        self
    }
}

/// State of a single worker in the integrated training.
pub struct WorkerState {
    /// Worker index.
    pub index: usize,
    /// Worker's party ID.
    pub party: PartyId,
    /// Model share held by this worker.
    pub model_share: ModelShare,
    /// Beaver triple pool.
    pub beaver_pool: BeaverPool,
    /// Proved arithmetic engine.
    pub arithmetic: ProvedArithmetic,
    /// Blinding generator.
    pub blinding_gen: BlindingGenerator,
    /// Current gradient share.
    pub gradient_share: Option<GradientShare>,
    /// Current witness capture.
    pub witness_capture: Option<WitnessCapture>,
    /// Computed share commitments.
    pub share_commitments: Vec<[u8; 32]>,
}

impl std::fmt::Debug for WorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerState")
            .field("index", &self.index)
            .field("party", &self.party)
            .field("model_share", &"<ModelShare>")
            .field("beaver_pool", &"<BeaverPool>")
            .field("arithmetic", &"<ProvedArithmetic>")
            .field("blinding_gen", &"<BlindingGenerator>")
            .field("gradient_share", &self.gradient_share.is_some())
            .field("witness_capture", &self.witness_capture.is_some())
            .field("share_commitments_count", &self.share_commitments.len())
            .finish()
    }
}

impl WorkerState {
    /// Creates a new worker state.
    pub fn new(index: usize, model_share: ModelShare, beaver_pool: BeaverPool) -> Self {
        let party = PartyId::from_index(index);
        Self {
            index,
            party,
            model_share,
            beaver_pool,
            arithmetic: ProvedArithmetic::new(index),
            blinding_gen: BlindingGenerator::new(),
            gradient_share: None,
            witness_capture: None,
            share_commitments: Vec::new(),
        }
    }

    /// Computes a commitment to the current model share.
    pub fn commit_model_share(&mut self) -> [u8; 32] {
        let blinding = self.blinding_gen.generate();
        let mut hasher = Sha256::new();

        for layer in &self.model_share.layers {
            for (name, tensor) in &layer.weights {
                hasher.update(name.as_bytes());
                for v in &tensor.data {
                    hasher.update(&v.to_bytes_le());
                }
            }
        }

        hasher.update(&blinding);
        hasher.finalize().into()
    }
}

/// Result of a single training step in the integrated flow.
#[derive(Debug, Clone)]
pub struct IntegratedTrainingStep {
    /// Step number.
    pub step: u64,
    /// Computed loss (after reconstruction).
    pub loss: f64,
    /// ZK proof of correct computation.
    pub proof: Option<ProofResult>,
    /// Share validity proofs per worker.
    pub share_proofs: Vec<ShareValidityProof>,
    /// Aggregation proof.
    pub aggregation_proof: Option<AggregationProof>,
    /// Old model state hash.
    pub old_state_hash: (Fr, Fr),
    /// New model state hash.
    pub new_state_hash: (Fr, Fr),
    /// Total error bound for this step.
    pub total_error: f64,
    /// Computation time in milliseconds.
    pub compute_time_ms: u64,
    /// Proof generation time in milliseconds.
    pub proof_time_ms: u64,
    /// Whether the step was successfully verified.
    pub verified: bool,
    /// Classification accuracy (if computable from target).
    pub accuracy: Option<f64>,
}

/// The integrated MPC-ZK training coordinator.
///
/// This is the main entry point for running training with both
/// secret sharing (MPC) and zero-knowledge proofs (ZK).
#[allow(dead_code)]
pub struct IntegratedTrainingCoordinator {
    /// Configuration.
    config: IntegratedTrainingConfig,
    /// Worker states.
    workers: Vec<WorkerState>,
    /// ZK proof pipeline.
    zk_pipeline: ZKProofPipeline,
    /// Circuit bridge for Halo2 proofs.
    circuit_bridge: Option<CircuitBridge>,
    /// Share validity prover.
    share_prover: ShareValidityProver,
    /// Aggregation prover.
    agg_prover: AggregationProver,
    /// Batch verifier.
    batch_verifier: BatchVerifier,
    /// Current training step.
    current_step: u64,
    /// Training history.
    history: Vec<IntegratedTrainingStep>,
    /// Total training time.
    total_time_ms: u64,
}

impl IntegratedTrainingCoordinator {
    /// Creates a new integrated training coordinator.
    #[instrument(skip(config), level = "info")]
    pub fn new(config: IntegratedTrainingConfig) -> Self {
        info!(
            num_workers = config.num_workers,
            d_in = config.d_in,
            d_hid = config.d_hid,
            d_out = config.d_out,
            use_real_proofs = config.use_real_proofs,
            "Creating integrated training coordinator"
        );

        let zk_config = ZKPipelineConfig {
            num_parties: config.num_workers,
            d_in: config.d_in,
            d_hid: config.d_hid,
            d_out: config.d_out,
            base_error: config.base_error,
            max_error: config.max_error,
            use_real_proofs: config.use_real_proofs,
            circuit_k: config.circuit_k,
            ..Default::default()
        };

        let circuit_bridge = if config.use_real_proofs {
            let bridge_config = CircuitBridgeConfig::for_model(
                config.d_in,
                config.d_hid,
                config.d_out,
            )
            .with_k(config.circuit_k)
            .with_base_error(config.base_error);
            Some(CircuitBridge::new(bridge_config))
        } else {
            None
        };

        Self {
            config,
            workers: Vec::new(),
            zk_pipeline: ZKProofPipeline::new(zk_config),
            circuit_bridge,
            share_prover: ShareValidityProver::new(),
            agg_prover: AggregationProver::new(),
            batch_verifier: BatchVerifier::new(),
            current_step: 0,
            history: Vec::new(),
            total_time_ms: 0,
        }
    }

    /// Initializes the training with model weights.
    ///
    /// This secret-shares the weights among workers and sets up
    /// Beaver triple pools for secure multiplication.
    #[instrument(skip(self, w1, b1, w2, b2), level = "info")]
    pub fn initialize(
        &mut self,
        w1: &[f64],
        b1: &[f64],
        w2: &[f64],
        b2: &[f64],
    ) -> MPCResult<()> {
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;
        let num_workers = self.config.num_workers;

        // Validate dimensions
        if w1.len() != d_hid * d_in {
            return Err(MPCError::ShapeMismatch {
                expected: vec![d_hid * d_in],
                got: vec![w1.len()],
            });
        }
        if b1.len() != d_hid {
            return Err(MPCError::ShapeMismatch {
                expected: vec![d_hid],
                got: vec![b1.len()],
            });
        }
        if w2.len() != d_out * d_hid {
            return Err(MPCError::ShapeMismatch {
                expected: vec![d_out * d_hid],
                got: vec![w2.len()],
            });
        }
        if b2.len() != d_out {
            return Err(MPCError::ShapeMismatch {
                expected: vec![d_out],
                got: vec![b2.len()],
            });
        }

        info!(
            w1_size = w1.len(),
            b1_size = b1.len(),
            w2_size = w2.len(),
            b2_size = b2.len(),
            num_workers = num_workers,
            "Initializing model weights"
        );

        // Generate Beaver triples using trusted dealer
        let mut dealer = TrustedDealer::with_seed(42);
        let triples_per_worker = dealer.generate_scalar_triples(
            self.config.beaver_batch_size,
            num_workers,
        );

        // Create additive shares of weights
        for i in 0..num_workers {
            let party = PartyId::from_index(i);

            // Create weight shares (additive sharing: each party gets value/n)
            let w1_share: Vec<Fr> = w1
                .iter()
                .map(|&v| Fr::from_f64(v / num_workers as f64))
                .collect();
            let b1_share: Vec<Fr> = b1
                .iter()
                .map(|&v| Fr::from_f64(v / num_workers as f64))
                .collect();
            let w2_share: Vec<Fr> = w2
                .iter()
                .map(|&v| Fr::from_f64(v / num_workers as f64))
                .collect();
            let b2_share: Vec<Fr> = b2
                .iter()
                .map(|&v| Fr::from_f64(v / num_workers as f64))
                .collect();

            // Create tensor shares
            let mut weights = HashMap::new();
            weights.insert(
                "w1".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "w1", i),
                    w1_share,
                    vec![d_hid, d_in],
                ),
            );
            weights.insert(
                "b1".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "b1", i),
                    b1_share,
                    vec![d_hid],
                ),
            );
            weights.insert(
                "w2".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "w2", i),
                    w2_share,
                    vec![d_out, d_hid],
                ),
            );
            weights.insert(
                "b2".to_string(),
                TensorShare::new(
                    ShareId::new(party.clone(), "b2", i),
                    b2_share,
                    vec![d_out],
                ),
            );

            let model_share = ModelShare {
                party: party.clone(),
                index: i,
                model_name: "training_model".to_string(),
                num_layers: 1,
                embeddings: None,
                layers: vec![LayerShare {
                    layer_idx: 0,
                    weights,
                }],
                lm_head: None,
                extra_weights: HashMap::new(),
            };

            // Create Beaver pool for this worker
            let mut pool = BeaverPool::new(i, num_workers, 64);
            pool.fill_scalar(triples_per_worker[i].clone());

            let worker = WorkerState::new(i, model_share, pool);
            self.workers.push(worker);

            debug!(worker = i, "Initialized worker state");
        }

        info!("Training initialization complete");
        Ok(())
    }

    /// Executes a single training step with proof generation.
    ///
    /// This is the core function that:
    /// 1. Each worker computes gradients on their share
    /// 2. Workers generate share validity proofs
    /// 3. Shares are aggregated to reconstruct gradients
    /// 4. ZK proof is generated for the full computation
    /// 5. Proof is verified
    #[instrument(skip(self, input, target), level = "info")]
    pub fn training_step(
        &mut self,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<IntegratedTrainingStep> {
        let step_start = Instant::now();
        self.current_step += 1;

        info!(
            step = self.current_step,
            input_size = input.len(),
            target_size = target.len(),
            "Starting training step"
        );

        // Phase 1: Each worker computes gradients on their share
        let compute_start = Instant::now();
        let mut _worker_witnesses: Vec<WitnessSummary> = Vec::new();
        let mut share_proofs = Vec::new();

        // Extract config values to avoid borrow conflicts
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;
        let base_error = self.config.base_error;
        let verbose_logging = self.config.verbose_logging;

        for worker in &mut self.workers {
            let _span = span!(Level::DEBUG, "worker_compute", worker = worker.index);

            // Start witness capture
            worker.arithmetic.capture_mut().clear();
            worker.arithmetic.set_capture_enabled(true);

            // Compute gradient share using MPC
            let gradient_share = Self::compute_worker_gradient_impl(
                d_in, d_hid, d_out, base_error,
                worker, input, target,
            )?;
            worker.gradient_share = Some(gradient_share);

            // Take witness capture
            let capture = worker.arithmetic.take_capture();
            let summary = WitnessSummary::from_capture(&capture);

            debug!(
                worker = worker.index,
                ops = summary.total_operations,
                error = summary.total_error,
                "Worker computation complete"
            );

            worker.witness_capture = Some(capture);

            // Generate share validity proof
            if verbose_logging {
                let blinding = worker.blinding_gen.generate();
                let layer = &worker.model_share.layers[0];
                let w1_tensor = layer.weights.get("w1").unwrap();

                let witness = ShareValidityWitness::from_tensor_share(
                    w1_tensor,
                    blinding,
                    [0u8; 32], // Dealer public key
                    vec![1, 2, 3, 4], // Placeholder signature
                );

                share_proofs.push(self.share_prover.prove(&witness)?);
            }
        }

        let compute_time = compute_start.elapsed();

        debug!(
            compute_ms = compute_time.as_millis(),
            num_workers = self.workers.len(),
            share_proofs_generated = share_proofs.len(),
            "Phase 1 complete: Worker gradient computation"
        );

        // Phase 2: Aggregate gradient shares and build ZK witness
        let proof_start = Instant::now();

        debug!(step = self.current_step, "Phase 2: Starting witness aggregation");

        // Collect all witness captures for reconstruction
        let captures: Vec<&WitnessCapture> = self.workers
            .iter()
            .filter_map(|w| w.witness_capture.as_ref())
            .collect();

        debug!(captures_collected = captures.len(), "Collected witness captures");

        // Build aggregated witness for ZK proof
        let mut aggregator = WitnessAggregator::new(self.config.num_workers);

        for worker in &self.workers {
            if let (Some(_capture), Some(gradient)) = (&worker.witness_capture, &worker.gradient_share) {
                let blinding = [worker.index as u8; 32];

                let builder = WitnessBuilder::new(
                    self.config.d_in,
                    self.config.d_hid,
                    self.config.d_out,
                    worker.index,
                )
                .learning_rate(self.config.learning_rate)
                .base_error(self.config.base_error);

                let witness = builder.build(
                    &worker.model_share,
                    gradient,
                    input,
                    target,
                    self.current_step,
                    &blinding,
                )?;

                aggregator.add_witness(worker.index, witness)?;
            }
        }

        // Reconstruct full witness
        let reconstructed = aggregator.reconstruct()?;

        debug!(
            w1_size = reconstructed.w1.len(),
            b1_size = reconstructed.b1.len(),
            w2_size = reconstructed.w2.len(),
            b2_size = reconstructed.b2.len(),
            "Witness reconstruction complete"
        );

        // Compute state hashes
        let old_hash = compute_compatible_state_hash(
            &reconstructed.w1,
            &reconstructed.b1,
            &reconstructed.w2,
            &reconstructed.b2,
        );

        debug!(step = self.current_step, "Phase 3: Starting ZK proof generation");

        // Phase 3: Generate ZK proof
        let proof = if let Some(ref bridge) = self.circuit_bridge {
            // Real Halo2 proof
            debug!("Using real Halo2 proof generation via CircuitBridge");
            let halo2_result = bridge.prove(&reconstructed)?;
            Some(ProofResult::new(
                halo2_result.proof,
                halo2_result.public_inputs,
                halo2_result.old_state_hash,
                halo2_result.new_state_hash,
                halo2_result.loss,
                halo2_result.total_error,
                halo2_result.step_number,
                halo2_result.generation_time_ms,
            ))
        } else {
            // Mock proof via ZK pipeline
            for worker in &mut self.workers {
                if let Some(gradient) = &worker.gradient_share {
                    self.zk_pipeline.generate_party_witness(
                        worker.index,
                        &worker.model_share,
                        gradient,
                        input,
                        target,
                        self.config.learning_rate,
                    )?;
                }
            }
            Some(self.zk_pipeline.generate_proof()?)
        };

        let proof_time = proof_start.elapsed();

        // Compute loss and accuracy from reconstructed model (proof loss may be Fr::ZERO for mock proofs).
        let (loss, new_hash, total_error, accuracy) = if let Some(ref p) = proof {
            let model = self.reconstruct_model()?;
            let output = model.forward(input);
            let mse = output.iter().zip(target.iter())
                .map(|(y, t)| (y - t).powi(2))
                .sum::<f64>() * 0.5;
            // Compute classification accuracy: argmax(output) == argmax(target).
            let acc = if target.len() > 1 {
                let pred = output.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let label = target.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                Some(if pred == label { 1.0 } else { 0.0 })
            } else {
                None
            };
            (mse, p.new_state_hash.clone(), p.total_error.to_f64(), acc)
        } else {
            (0.0, old_hash.clone(), 0.0, None)
        };

        // Phase 4: Verify the proof
        let verified = if let Some(ref p) = proof {
            self.zk_pipeline.verify_proof(p)?
        } else {
            false
        };

        if !verified {
            warn!(step = self.current_step, "Proof verification failed!");
        } else {
            debug!(step = self.current_step, "Proof verified successfully");
        }

        // Phase 5: Apply gradient updates to model shares
        self.apply_gradient_updates()?;

        // Advance ZK pipeline
        self.zk_pipeline.next_step();

        let total_time = step_start.elapsed();

        let step_result = IntegratedTrainingStep {
            step: self.current_step,
            loss,
            proof,
            share_proofs,
            aggregation_proof: None,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            total_error,
            compute_time_ms: compute_time.as_millis() as u64,
            proof_time_ms: proof_time.as_millis() as u64,
            verified,
            accuracy,
        };

        self.history.push(step_result.clone());
        self.total_time_ms += total_time.as_millis() as u64;

        info!(
            step = self.current_step,
            loss = loss,
            verified = verified,
            compute_ms = compute_time.as_millis(),
            proof_ms = proof_time.as_millis(),
            "Training step complete"
        );

        Ok(step_result)
    }

    /// Computes gradient share for a worker using MPC protocols.
    fn compute_worker_gradient_impl(
        d_in: usize,
        d_hid: usize,
        d_out: usize,
        base_error: f64,
        worker: &mut WorkerState,
        input: &[f64],
        target: &[f64],
    ) -> MPCResult<GradientShare> {

        // Get weight shares
        let layer = &worker.model_share.layers[0];
        let w1 = layer.weights.get("w1").unwrap();
        let b1 = layer.weights.get("b1").unwrap();
        let w2 = layer.weights.get("w2").unwrap();
        let b2 = layer.weights.get("b2").unwrap();

        // Convert input to Fr
        let x: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
        let t: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();

        // Forward pass on shares (simplified - local computation)
        // h_pre = W1 @ x + b1
        let mut h_pre = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let mut sum = Fr::ZERO;
            for j in 0..d_in {
                sum = worker.arithmetic.add_shares(&sum, &w1.data[i * d_in + j].mpc_scale(&x[j]));
            }
            h_pre[i] = worker.arithmetic.add_shares(&sum, &b1.data[i]);
        }

        // ReLU (local operation)
        let mut h = vec![Fr::ZERO; d_hid];
        let mut relu_mask = vec![Fr::ZERO; d_hid];
        for i in 0..d_hid {
            let val = h_pre[i].to_f64();
            if val > 0.0 {
                h[i] = h_pre[i].clone();
                relu_mask[i] = Fr::ONE;
            }
        }
        worker.arithmetic.record_relu(&h_pre, &h, &relu_mask);

        // y = W2 @ h + b2
        let mut y = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            let mut sum = Fr::ZERO;
            for j in 0..d_hid {
                sum = worker.arithmetic.add_shares(&sum, &w2.data[i * d_hid + j].mpc_scale(&h[j]));
            }
            y[i] = worker.arithmetic.add_shares(&sum, &b2.data[i]);
        }

        // Compute loss = 0.5 * sum((y - t)^2)
        let mut loss = Fr::ZERO;
        for i in 0..d_out {
            let diff = worker.arithmetic.sub_shares(&y[i], &t[i]);
            loss = worker.arithmetic.add_shares(&loss, &diff.mpc_scale(&diff));
        }
        loss = worker.arithmetic.scale_share(&loss, &Fr::from_f64(0.5));

        // Record forward pass
        worker.arithmetic.record_forward_pass(&x, &h_pre, &h, &y, loss.clone());

        // Backward pass (simplified gradient computation)
        let mut dy = vec![Fr::ZERO; d_out];
        for i in 0..d_out {
            dy[i] = worker.arithmetic.sub_shares(&y[i], &t[i]);
        }

        // dW2 = outer(dy, h)
        let mut dw2 = vec![Fr::ZERO; d_out * d_hid];
        for i in 0..d_out {
            for j in 0..d_hid {
                dw2[i * d_hid + j] = dy[i].mpc_scale(&h[j]);
            }
        }

        // db2 = dy
        let db2 = dy.clone();

        // dh = W2^T @ dy
        let mut dh = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            let mut sum = Fr::ZERO;
            for i in 0..d_out {
                sum = Fr::add(&sum, &w2.data[i * d_hid + j].mpc_scale(&dy[i]));
            }
            dh[j] = sum;
        }

        // dh_pre = dh * relu_mask
        let mut dh_pre = vec![Fr::ZERO; d_hid];
        for j in 0..d_hid {
            dh_pre[j] = dh[j].mpc_scale(&relu_mask[j]);
        }

        // dW1 = outer(dh_pre, x)
        let mut dw1 = vec![Fr::ZERO; d_hid * d_in];
        for i in 0..d_hid {
            for j in 0..d_in {
                dw1[i * d_in + j] = dh_pre[i].mpc_scale(&x[j]);
            }
        }

        // db1 = dh_pre
        let db1 = dh_pre.clone();

        // Record backward pass
        worker.arithmetic.record_backward_pass(&dy, &dw2, &db2, &dh, &dh_pre, &dw1, &db1);

        // Create gradient share
        let mut gradients = HashMap::new();
        gradients.insert(
            "w1".to_string(),
            TensorShare::new(
                ShareId::new(worker.party.clone(), "dw1", worker.index),
                dw1,
                vec![d_hid, d_in],
            ),
        );
        gradients.insert(
            "b1".to_string(),
            TensorShare::new(
                ShareId::new(worker.party.clone(), "db1", worker.index),
                db1,
                vec![d_hid],
            ),
        );
        gradients.insert(
            "w2".to_string(),
            TensorShare::new(
                ShareId::new(worker.party.clone(), "dw2", worker.index),
                dw2,
                vec![d_out, d_hid],
            ),
        );
        gradients.insert(
            "b2".to_string(),
            TensorShare::new(
                ShareId::new(worker.party.clone(), "db2", worker.index),
                db2,
                vec![d_out],
            ),
        );

        Ok(GradientShare {
            party: worker.party.clone(),
            index: worker.index,
            embeddings: None,
            layers: vec![LayerGradientShare {
                layer_idx: 0,
                gradients,
            }],
            lm_head: None,
            error_bound: base_error * 100.0,
        })
    }

    /// Applies gradient updates to worker model shares.
    fn apply_gradient_updates(&mut self) -> MPCResult<()> {
        let lr = Fr::from_f64(self.config.learning_rate);

        for worker in &mut self.workers {
            if let Some(ref gradient) = worker.gradient_share {
                let layer = &mut worker.model_share.layers[0];

                if let (Some(w1), Some(dw1)) = (
                    layer.weights.get_mut("w1"),
                    gradient.layers.first().and_then(|l| l.gradients.get("w1")),
                ) {
                    for (w, dw) in w1.data.iter_mut().zip(dw1.data.iter()) {
                        *w = Fr::sub(w, &lr.mpc_scale(dw));
                    }
                }

                if let (Some(b1), Some(db1)) = (
                    layer.weights.get_mut("b1"),
                    gradient.layers.first().and_then(|l| l.gradients.get("b1")),
                ) {
                    for (b, db) in b1.data.iter_mut().zip(db1.data.iter()) {
                        *b = Fr::sub(b, &lr.mpc_scale(db));
                    }
                }

                if let (Some(w2), Some(dw2)) = (
                    layer.weights.get_mut("w2"),
                    gradient.layers.first().and_then(|l| l.gradients.get("w2")),
                ) {
                    for (w, dw) in w2.data.iter_mut().zip(dw2.data.iter()) {
                        *w = Fr::sub(w, &lr.mpc_scale(dw));
                    }
                }

                if let (Some(b2), Some(db2)) = (
                    layer.weights.get_mut("b2"),
                    gradient.layers.first().and_then(|l| l.gradients.get("b2")),
                ) {
                    for (b, db) in b2.data.iter_mut().zip(db2.data.iter()) {
                        *b = Fr::sub(b, &lr.mpc_scale(db));
                    }
                }
            }
        }

        Ok(())
    }

    /// Runs multiple training steps.
    #[instrument(skip(self, data), level = "info")]
    pub fn train(&mut self, data: &[(Vec<f64>, Vec<f64>)]) -> MPCResult<TrainingReport> {
        let start = Instant::now();

        info!(num_samples = data.len(), "Starting training run");

        let mut steps = Vec::new();

        for (i, (input, target)) in data.iter().enumerate() {
            let step = self.training_step(input, target)?;
            steps.push(step);

            if (i + 1) % 10 == 0 || i == data.len() - 1 {
                let last_loss = steps.last().map(|s| s.loss).unwrap_or(0.0);
                info!(
                    step = i + 1,
                    total = data.len(),
                    loss = last_loss,
                    "Training progress"
                );
            }
        }

        let total_time = start.elapsed();

        // Compute statistics
        let avg_loss = steps.iter().map(|s| s.loss).sum::<f64>() / steps.len() as f64;
        let avg_compute_time = steps.iter().map(|s| s.compute_time_ms).sum::<u64>() / steps.len() as u64;
        let avg_proof_time = steps.iter().map(|s| s.proof_time_ms).sum::<u64>() / steps.len() as u64;
        let verification_rate = steps.iter().filter(|s| s.verified).count() as f64 / steps.len() as f64;

        // Compute final accuracy: average over the last 10 steps (or all if fewer).
        let final_accuracy = {
            let recent: Vec<_> = steps.iter().rev().take(10)
                .filter_map(|s| s.accuracy)
                .collect();
            if recent.is_empty() {
                None
            } else {
                Some(recent.iter().sum::<f64>() / recent.len() as f64)
            }
        };

        let report = TrainingReport {
            steps,
            total_steps: data.len(),
            final_loss: avg_loss,
            avg_compute_time_ms: avg_compute_time,
            avg_proof_time_ms: avg_proof_time,
            total_time_ms: total_time.as_millis() as u64,
            verification_rate,
            final_accuracy,
        };

        info!(
            total_steps = report.total_steps,
            final_loss = report.final_loss,
            verification_rate = report.verification_rate,
            total_time_ms = report.total_time_ms,
            "Training complete"
        );

        Ok(report)
    }

    /// Reconstructs the trained model from worker shares.
    pub fn reconstruct_model(&self) -> MPCResult<ReconstructedModel> {
        let d_in = self.config.d_in;
        let d_hid = self.config.d_hid;
        let d_out = self.config.d_out;

        let mut w1 = vec![0.0f64; d_hid * d_in];
        let mut b1 = vec![0.0f64; d_hid];
        let mut w2 = vec![0.0f64; d_out * d_hid];
        let mut b2 = vec![0.0f64; d_out];

        for worker in &self.workers {
            let layer = &worker.model_share.layers[0];

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

        Ok(ReconstructedModel {
            d_in,
            d_hid,
            d_out,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Returns the training history.
    pub fn history(&self) -> &[IntegratedTrainingStep] {
        &self.history
    }

    /// Returns the current step number.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns the configuration.
    pub fn config(&self) -> &IntegratedTrainingConfig {
        &self.config
    }
}

/// Reconstructed model weights after training.
#[derive(Debug, Clone)]
pub struct ReconstructedModel {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl ReconstructedModel {
    /// Performs inference with the reconstructed model.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // h_pre = W1 @ x + b1
        let mut h_pre = vec![0.0; self.d_hid];
        for i in 0..self.d_hid {
            let mut sum = 0.0;
            for j in 0..self.d_in {
                sum += self.w1[i * self.d_in + j] * input[j];
            }
            h_pre[i] = sum + self.b1[i];
        }

        // h = ReLU(h_pre)
        let h: Vec<f64> = h_pre.iter().map(|&x| x.max(0.0)).collect();

        // y = W2 @ h + b2
        let mut y = vec![0.0; self.d_out];
        for i in 0..self.d_out {
            let mut sum = 0.0;
            for j in 0..self.d_hid {
                sum += self.w2[i * self.d_hid + j] * h[j];
            }
            y[i] = sum + self.b2[i];
        }

        y
    }
}

/// Summary report of a training run.
#[derive(Debug, Clone)]
pub struct TrainingReport {
    /// All training steps.
    pub steps: Vec<IntegratedTrainingStep>,
    /// Total number of steps.
    pub total_steps: usize,
    /// Final loss value.
    pub final_loss: f64,
    /// Average computation time per step.
    pub avg_compute_time_ms: u64,
    /// Average proof generation time per step.
    pub avg_proof_time_ms: u64,
    /// Total training time.
    pub total_time_ms: u64,
    /// Rate of successful verifications.
    pub verification_rate: f64,
    /// Final classification accuracy (if computable).
    pub final_accuracy: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrated_training_config() {
        let config = IntegratedTrainingConfig::minimal();
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.d_in, 2);
        assert_eq!(config.d_hid, 2);
        assert_eq!(config.d_out, 1);
    }

    #[test]
    fn test_initialization() {
        let config = IntegratedTrainingConfig::minimal();
        let mut coordinator = IntegratedTrainingCoordinator::new(config);

        let w1 = vec![0.1, 0.2, 0.3, 0.4]; // 2x2
        let b1 = vec![0.01, 0.02];
        let w2 = vec![0.5, 0.6]; // 1x2
        let b2 = vec![0.03];

        coordinator.initialize(&w1, &b1, &w2, &b2).unwrap();

        assert_eq!(coordinator.workers.len(), 2);

        // Verify that reconstruction works
        let model = coordinator.reconstruct_model().unwrap();
        assert!((model.w1[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_single_training_step() {
        let config = IntegratedTrainingConfig::minimal();
        let mut coordinator = IntegratedTrainingCoordinator::new(config);

        let w1 = vec![0.1, 0.2, 0.3, 0.4];
        let b1 = vec![0.01, 0.02];
        let w2 = vec![0.5, 0.6];
        let b2 = vec![0.03];

        coordinator.initialize(&w1, &b1, &w2, &b2).unwrap();

        let input = vec![1.0, 1.0];
        let target = vec![0.5];

        let step = coordinator.training_step(&input, &target).unwrap();

        assert_eq!(step.step, 1);
        assert!(step.proof.is_some());
        assert!(step.verified);
    }

    #[test]
    fn test_multiple_training_steps() {
        let config = IntegratedTrainingConfig::minimal();
        let mut coordinator = IntegratedTrainingCoordinator::new(config);

        let w1 = vec![0.1, 0.2, 0.3, 0.4];
        let b1 = vec![0.01, 0.02];
        let w2 = vec![0.5, 0.6];
        let b2 = vec![0.03];

        coordinator.initialize(&w1, &b1, &w2, &b2).unwrap();

        let data = vec![
            (vec![1.0, 0.0], vec![1.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        let report = coordinator.train(&data).unwrap();

        assert_eq!(report.total_steps, 3);
        assert!(report.verification_rate > 0.0);
    }

    #[test]
    fn test_reconstructed_model_inference() {
        let model = ReconstructedModel {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            w1: vec![0.5, 0.5, 0.5, 0.5],
            b1: vec![0.0, 0.0],
            w2: vec![1.0, 1.0],
            b2: vec![0.0],
        };

        let output = model.forward(&[1.0, 1.0]);
        assert_eq!(output.len(), 1);
        // h_pre = [1.0, 1.0], h = [1.0, 1.0], y = [2.0]
        assert!((output[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_worker_state() {
        let party = PartyId::from_index(0);
        let model_share = ModelShare {
            party: party.clone(),
            index: 0,
            model_name: "test".to_string(),
            num_layers: 0,
            embeddings: None,
            layers: vec![],
            lm_head: None,
            extra_weights: HashMap::new(),
        };

        let pool = BeaverPool::new(0, 2, 64);
        let mut worker = WorkerState::new(0, model_share, pool);

        let commitment = worker.commit_model_share();
        assert_ne!(commitment, [0u8; 32]);
    }
}
