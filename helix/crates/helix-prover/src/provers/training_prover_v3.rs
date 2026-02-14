//! ML Training Step V3 Prover (N-Layer MLP).
//!
//! Wraps the `MLTrainingStepV3Circuit` from `helix-circuits` with the
//! Halo2 `ProverPipeline` to produce real KZG proofs for N-layer MLP
//! training steps.
//!
//! This prover supports arbitrary depth networks with configurable
//! activations (ReLU, Tanh, Identity) and loss functions (MSE, CrossEntropy).
//! The proof format is identical to V2 — same 8 public inputs, same
//! EVM-compatible proof encoding — so on-chain contracts need no changes.

use crate::cache::witness_cache::{
    SharedWitnessCache, WitnessCachedProof, WitnessHash, WitnessHashBuilder, shared_witness_cache,
};
use crate::keys::generate_keys_for_circuit;
use crate::pipeline::{
    CancellationToken, PipelineConfig, PipelineError, ProofPhase, ProofProgress, ProgressCallback,
    ProverPipeline, RetryConfig, no_progress_callback,
};
use crate::provers::training_prover_v2::{
    TrainingProofResultV2, TrainingProverError, TrainingProverResult, EvmProofBundle,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use helix_circuits::ml::config::MLPArchitecture;
use helix_circuits::ml::training_step_v2::NUM_PUBLIC_INPUTS;
use helix_circuits::ml::training_step_v3::{
    compute_state_hash_v3, compute_witness_v3, create_zero_v3_witness,
    MLTrainingStepV3Circuit, MLTrainingStepV3Witness,
};
use helix_circuits::verifier::{
    SolidityGenerator, VkData, ProofFormatError,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, instrument, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the V3 N-layer prover.
#[derive(Debug, Clone)]
pub struct V3ProverConfig {
    /// K parameter (circuit size = 2^k rows).
    pub k: u32,
    /// ReLU lookup range half-width.
    pub relu_range: usize,
    /// Tanh lookup range half-width.
    pub tanh_range: usize,
    /// Exp lookup range.
    pub exp_range: usize,
    /// Exp lookup scale.
    pub exp_scale: u64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
    /// Base error per operation.
    pub base_error: Fr,
    /// Whether to self-verify proofs.
    pub self_verify: bool,
    /// Retry configuration.
    pub retry: RetryConfig,
    /// Proof generation timeout.
    pub proof_timeout: Option<Duration>,
    /// Verification timeout.
    pub verify_timeout: Option<Duration>,
    /// Seed for deterministic proof generation.
    pub deterministic_seed: Option<[u8; 32]>,
    /// Whether to use witness caching.
    pub use_witness_cache: bool,
    /// Enable detailed tracing.
    pub enable_tracing: bool,
}

impl Default for V3ProverConfig {
    fn default() -> Self {
        Self {
            k: 14,
            relu_range: 128,
            tanh_range: 128,
            exp_range: 256,
            exp_scale: 1000,
            use_freivalds: true,
            base_error: Fr::from(1u64),
            self_verify: true,
            retry: RetryConfig::default(),
            proof_timeout: Some(Duration::from_secs(300)),
            verify_timeout: Some(Duration::from_secs(30)),
            deterministic_seed: None,
            use_witness_cache: true,
            enable_tracing: true,
        }
    }
}

impl V3ProverConfig {
    /// Creates a minimal configuration for testing.
    pub fn minimal() -> Self {
        Self {
            k: 12,
            relu_range: 64,
            tanh_range: 64,
            exp_range: 128,
            self_verify: false,
            retry: RetryConfig::none(),
            use_witness_cache: false,
            enable_tracing: false,
            ..Default::default()
        }
    }

    /// Creates a production configuration.
    pub fn production() -> Self {
        Self {
            self_verify: true,
            retry: RetryConfig::aggressive(),
            use_witness_cache: true,
            enable_tracing: true,
            ..Default::default()
        }
    }

    /// Sets the K parameter.
    pub fn with_k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Enables deterministic proof generation.
    pub fn deterministic(mut self, seed: [u8; 32]) -> Self {
        self.deterministic_seed = Some(seed);
        self
    }
}

// ============================================================================
// ML Training Prover V3
// ============================================================================

/// N-layer MLP Training Step Prover.
///
/// Generates Halo2 KZG proofs for arbitrary-depth MLP training steps.
/// Reuses `TrainingProofResultV2` and `EvmProofBundle` since the proof
/// format is identical (same 8 public inputs, same EVM encoding).
pub struct MLTrainingProverV3 {
    /// Halo2 proving pipeline.
    pipeline: ProverPipeline<MLTrainingStepV3Circuit>,
    /// Configuration.
    config: V3ProverConfig,
    /// Network architecture.
    arch: MLPArchitecture,
    /// Witness cache.
    cache: Option<SharedWitnessCache>,
    /// Proof counter.
    proof_count: AtomicU64,
    /// Cache hit counter.
    cache_hits: AtomicU64,
}

impl MLTrainingProverV3 {
    /// Creates a new V3 prover for the given architecture.
    pub fn new(arch: MLPArchitecture) -> Self {
        Self::with_config(arch, V3ProverConfig::default())
    }

    /// Creates a V3 prover with custom configuration.
    #[instrument(skip_all, fields(num_layers = arch.num_layers(), k = config.k))]
    pub fn with_config(arch: MLPArchitecture, config: V3ProverConfig) -> Self {
        let pipeline_config = PipelineConfig {
            k: config.k,
            self_verify: false,
            retry: config.retry.clone(),
            proof_timeout: config.proof_timeout,
            verify_timeout: config.verify_timeout,
            deterministic_seed: config.deterministic_seed,
            enable_tracing: config.enable_tracing,
        };

        // Create dummy witness for key generation
        let dummy_witness = create_zero_v3_witness(&arch);
        let setup_circuit = MLTrainingStepV3Circuit::new(
            dummy_witness,
            config.relu_range,
            config.tanh_range,
            config.exp_range,
            config.use_freivalds,
        );

        let circuit_keys = generate_keys_for_circuit(
            &setup_circuit,
            "ml_training_step_v3",
            1,
            config.k,
        );

        let pipeline = ProverPipeline::from_keys(
            pipeline_config,
            circuit_keys.params,
            circuit_keys.pk,
            circuit_keys.vk,
        );

        let cache = if config.use_witness_cache {
            Some(shared_witness_cache())
        } else {
            None
        };

        Self {
            pipeline,
            config,
            arch,
            cache,
            proof_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        }
    }

    /// Returns the network architecture.
    pub fn arch(&self) -> &MLPArchitecture {
        &self.arch
    }

    /// Returns the configuration.
    pub fn config(&self) -> &V3ProverConfig {
        &self.config
    }

    /// Returns the number of proofs generated.
    pub fn proof_count(&self) -> u64 {
        self.proof_count.load(Ordering::Relaxed)
    }

    /// Returns the number of cache hits.
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Builds a witness from raw training data.
    pub fn build_witness(
        arch: &MLPArchitecture,
        x: &[Fr],
        target: &[Fr],
        layer_weights: &[(&[Fr], &[Fr])],
        lr: Fr,
        step_number: u64,
        base_error: Fr,
    ) -> MLTrainingStepV3Witness {
        let old_hash = compute_state_hash_v3(layer_weights);

        // First pass to compute new weights
        let tmp = compute_witness_v3(
            arch, x, target, layer_weights, lr,
            old_hash, (Fr::zero(), Fr::zero()), step_number, base_error,
        );

        let new_weights: Vec<(Vec<Fr>, Vec<Fr>)> = tmp.layers.iter()
            .map(|l| (l.weights_new.clone(), l.biases_new.clone()))
            .collect();
        let new_weight_refs: Vec<(&[Fr], &[Fr])> = new_weights.iter()
            .map(|(w, b)| (w.as_slice(), b.as_slice()))
            .collect();
        let new_hash = compute_state_hash_v3(&new_weight_refs);

        // Second pass with correct new hash
        compute_witness_v3(
            arch, x, target, layer_weights, lr,
            old_hash, new_hash, step_number, base_error,
        )
    }

    /// Computes the witness hash for caching.
    pub fn compute_witness_hash(witness: &MLTrainingStepV3Witness) -> WitnessHash {
        let mut builder = WitnessHashBuilder::new();
        // Hash all layer weights
        for lw in &witness.layers {
            builder = builder.add_field_elements(&lw.weights);
            builder = builder.add_field_elements(&lw.biases);
        }
        builder = builder.add_field_elements(&witness.x);
        builder = builder.add_field_elements(&witness.target);
        builder = builder.add_field_elements(&[witness.lr]);
        builder = builder.add_u64(witness.step_number);
        builder.finish()
    }

    /// Generates a proof for the given witness.
    #[instrument(skip_all, fields(step = witness.step_number))]
    pub fn prove(&self, witness: &MLTrainingStepV3Witness) -> TrainingProverResult<TrainingProofResultV2> {
        self.prove_with_options(witness, no_progress_callback(), None)
    }

    /// Generates a proof with progress callbacks and cancellation support.
    #[instrument(skip_all, fields(step = witness.step_number))]
    pub fn prove_with_options(
        &self,
        witness: &MLTrainingStepV3Witness,
        progress: ProgressCallback,
        cancel_token: Option<&CancellationToken>,
    ) -> TrainingProverResult<TrainingProofResultV2> {
        info!("Proof generation started for V3 training step");
        let start = Instant::now();

        if let Some(token) = cancel_token {
            if token.is_cancelled() {
                return Err(TrainingProverError::Cancelled);
            }
        }

        // Validate witness
        if let Err(e) = witness.validate() {
            return Err(TrainingProverError::WitnessValidation {
                message: e,
                field: "witness".to_string(),
            });
        }

        let witness_hash = Self::compute_witness_hash(witness);

        // Check cache
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&witness_hash) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.proof_count.fetch_add(1, Ordering::Relaxed);

                let pi = witness.public_inputs();
                return Ok(TrainingProofResultV2 {
                    proof: cached.proof,
                    public_inputs: pi,
                    loss: witness.loss,
                    total_error: witness.total_error,
                    step_number: witness.step_number,
                    old_state_hash: witness.old_state_hash,
                    new_state_hash: witness.new_state_hash,
                    verified: cached.verified,
                    generation_time: Duration::from_millis(cached.generation_time_ms),
                    verification_time: None,
                    attempts: 1,
                    from_cache: true,
                    witness_hash: Some(witness_hash),
                });
            }
        }

        // Build circuit
        let circuit = MLTrainingStepV3Circuit::new(
            witness.clone(),
            self.config.relu_range,
            self.config.tanh_range,
            self.config.exp_range,
            self.config.use_freivalds,
        );

        let pi = witness.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Generate proof
        progress(ProofProgress {
            phase: ProofPhase::Synthesis,
            phase_progress: 0.0,
            overall_progress: 0.2,
            elapsed: start.elapsed(),
            estimated_remaining: None,
            attempt: 0,
            message: Some("Generating V3 proof".to_string()),
        });

        let proof_result = self
            .pipeline
            .prove_with_options(&circuit, &pi_refs, progress.clone(), cancel_token)
            .map_err(|e| {
                error!(
                    step = witness.step_number,
                    error = %e,
                    "V3 proof generation failed"
                );
                TrainingProverError::Pipeline(e)
            })?;

        let gen_time = proof_result.generation_time;
        let proof = proof_result.proof;
        let attempts = proof_result.attempts;

        // Self-verification
        let (verified, verify_time) = if self.config.self_verify {
            let verify_start = Instant::now();
            let is_valid = self.pipeline.verify(&proof, &pi_refs)?;
            if !is_valid {
                return Err(TrainingProverError::SelfVerificationFailed);
            }
            if proof.is_empty() || proof.len() % 32 != 0 {
                return Err(TrainingProverError::EvmSerializationFailed(
                    ProofFormatError::TooShort { got: proof.len(), min: 32 },
                ));
            }
            (true, Some(verify_start.elapsed()))
        } else {
            (false, None)
        };

        // Store in cache
        if let Some(ref cache) = self.cache {
            let cached = WitnessCachedProof::new(
                witness_hash,
                proof.clone(),
                pi.iter()
                    .map(|f| {
                        let bytes = f.to_repr();
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(bytes.as_ref());
                        arr
                    })
                    .collect(),
                gen_time.as_millis() as u64,
            );
            cache.insert(cached);
            if verified {
                cache.mark_verified(&witness_hash);
            }
        }

        self.proof_count.fetch_add(1, Ordering::Relaxed);

        info!(
            step = witness.step_number,
            num_layers = self.arch.num_layers(),
            proof_size = proof.len(),
            generation_time_ms = gen_time.as_millis() as u64,
            verified,
            "Proof generation complete"
        );

        Ok(TrainingProofResultV2 {
            proof,
            public_inputs: pi,
            loss: witness.loss,
            total_error: witness.total_error,
            step_number: witness.step_number,
            old_state_hash: witness.old_state_hash,
            new_state_hash: witness.new_state_hash,
            verified,
            generation_time: gen_time,
            verification_time: verify_time,
            attempts,
            from_cache: false,
            witness_hash: Some(witness_hash),
        })
    }

    /// Verifies a proof against the given public inputs.
    #[instrument(skip_all, fields(proof_size = proof.len()))]
    pub fn verify(&self, proof: &[u8], public_inputs: &[Fr]) -> bool {
        let pi_refs: Vec<&[Fr]> = vec![public_inputs];
        match self.pipeline.verify(proof, &pi_refs) {
            Ok(true) => true,
            Ok(false) => {
                warn!("V3 proof verification failed");
                false
            }
            Err(e) => {
                error!(error = %e, "V3 proof verification error");
                false
            }
        }
    }

    /// Verifies a `TrainingProofResultV2`.
    pub fn verify_result(&self, result: &TrainingProofResultV2) -> bool {
        if result.proof.is_empty() {
            return false;
        }
        self.verify(&result.proof, &result.public_inputs)
    }

    /// Exports verification key data for contract deployment.
    #[instrument(skip_all)]
    pub fn export_vk_data(&self) -> Result<VkData, TrainingProverError> {
        let extracted = self
            .pipeline
            .extract_vk_data(NUM_PUBLIC_INPUTS)
            .ok_or_else(|| TrainingProverError::NotInitialized {
                message: "Pipeline VK not initialized".to_string(),
            })?;
        Ok(VkData {
            g1: extracted.g1,
            s_g2: extracted.s_g2,
            neg_g2: extracted.neg_g2,
            num_advices: extracted.num_advices,
        })
    }

    /// Generates a proof and exports everything needed for on-chain verification.
    #[instrument(skip_all, fields(step = witness.step_number))]
    pub fn prove_and_export_evm(
        &self,
        witness: &MLTrainingStepV3Witness,
    ) -> TrainingProverResult<EvmProofBundle> {
        let result = self.prove(witness)?;
        let evm_proof = result.to_evm_proof()
            .map_err(TrainingProverError::EvmSerializationFailed)?;
        let evm_public_inputs = result.to_evm_public_inputs();
        let vk_deployment_args = self.export_vk_data()?;

        Ok(EvmProofBundle {
            evm_proof,
            evm_public_inputs,
            vk_deployment_args,
            result,
        })
    }

    /// Clears the witness cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use helix_circuits::ml::config::{CircuitActivation, LossFunction};

    /// Standalone prove-and-verify test: keygen uses same circuit as proving.
    #[test]
    fn test_v3_raw_prove_verify() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use helix_circuits::halo2_proofs::{
            plonk::{keygen_vk, keygen_pk, create_proof, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            transcript::TranscriptWriterBuffer,
        };
        use helix_circuits::halo2curves::bn256::{Bn256, G1Affine};
        use halo2_solidity_verifier::Keccak256Transcript;
        use rand_core::OsRng;

        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let relu_range = 64;
        let tanh_range = 64;
        let exp_range = 128;
        let k = 12;

        // Build witness
        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::zero(), Fr::zero()];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::zero()];
        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];
        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];
        let witness = MLTrainingProverV3::build_witness(
            &arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64),
        );
        let pi = witness.public_inputs();

        let circuit = MLTrainingStepV3Circuit::new(
            witness, relu_range, tanh_range, exp_range, true,
        );

        // Keygen with SAME circuit
        use crate::keys::HELIX_SRS_SEED;
        use rand::SeedableRng;
        let srs_rng = rand::rngs::StdRng::from_seed(HELIX_SRS_SEED);
        let params = ParamsKZG::<Bn256>::setup(k, srs_rng);
        let vk = keygen_vk(&params, &circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &circuit).expect("keygen_pk failed");

        // Prove
        let instances = vec![pi.clone()];
        let mut transcript = Keccak256Transcript::new(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,_,_,_,
        >(&params, &pk, &[circuit], &[instances.clone()], OsRng, &mut transcript)
        .expect("create_proof failed");
        let proof = transcript.finalize();

        // Verify
        let mut verifier_transcript = Keccak256Transcript::new(proof.as_slice());
        let verifier_params = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,_,
            SingleStrategy<Bn256>,
        >(&verifier_params, &vk, &[instances], &mut verifier_transcript);
        assert!(verified, "raw V3 proof (same keygen circuit) must verify");
    }

    /// Test with SEPARATE keygen circuit (zero witness) and prove circuit (real witness).
    /// This mimics what MLTrainingProverV3 does.
    #[test]
    fn test_v3_separate_keygen_prove() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use helix_circuits::halo2_proofs::{
            plonk::{keygen_vk, keygen_pk, create_proof, verify_proof_multi},
            poly::kzg::{
                commitment::{KZGCommitmentScheme, ParamsKZG},
                multiopen::{ProverSHPLONK, VerifierSHPLONK},
                strategy::SingleStrategy,
            },
            transcript::TranscriptWriterBuffer,
        };
        use helix_circuits::halo2curves::bn256::{Bn256, G1Affine};
        use halo2_solidity_verifier::Keccak256Transcript;
        use rand_core::OsRng;

        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let relu_range = 64;
        let tanh_range = 64;
        let exp_range = 128;
        let k = 12;

        // KEYGEN with zero witness (same as wrapper)
        let dummy_witness = create_zero_v3_witness(&arch);
        let keygen_circuit = MLTrainingStepV3Circuit::new(
            dummy_witness, relu_range, tanh_range, exp_range, true,
        );

        use crate::keys::HELIX_SRS_SEED;
        use rand::SeedableRng;
        let srs_rng = rand::rngs::StdRng::from_seed(HELIX_SRS_SEED);
        let params = ParamsKZG::<Bn256>::setup(k, srs_rng);
        let vk = keygen_vk(&params, &keygen_circuit).expect("keygen_vk failed");
        let pk = keygen_pk(&params, vk.clone(), &keygen_circuit).expect("keygen_pk failed");
        eprintln!("Keygen with zero witness done");

        // PROVE with real witness
        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::zero(), Fr::zero()];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::zero()];
        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];
        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];
        let witness = MLTrainingProverV3::build_witness(
            &arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64),
        );
        let pi = witness.public_inputs();

        let prove_circuit = MLTrainingStepV3Circuit::new(
            witness, relu_range, tanh_range, exp_range, true,
        );

        let instances = vec![pi.clone()];
        let mut transcript = Keccak256Transcript::new(vec![]);
        create_proof::<
            KZGCommitmentScheme<Bn256>,
            ProverSHPLONK<'_, Bn256>,
            _,_,_,_,
        >(&params, &pk, &[prove_circuit], &[instances.clone()], OsRng, &mut transcript)
        .expect("create_proof failed");
        let proof = transcript.finalize();
        eprintln!("Proof with real witness: {} bytes", proof.len());

        // Verify
        let mut verifier_transcript = Keccak256Transcript::new(proof.as_slice());
        let verifier_params = params.verifier_params();
        let verified = verify_proof_multi::<
            KZGCommitmentScheme<Bn256>,
            VerifierSHPLONK<Bn256>,
            _,_,
            SingleStrategy<Bn256>,
        >(&verifier_params, &vk, &[instances], &mut verifier_transcript);

        eprintln!("Separate keygen/prove verify: {}", verified);
        assert!(verified, "V3 proof with separate keygen/prove must verify");
    }

    #[test]
    fn test_v3_prover_3layer() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let arch = MLPArchitecture::from_dims(
            &[2, 3, 2, 1],
            &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
            LossFunction::MSE,
        );

        // Use k=14 to have enough rows for 3-layer network
        let config = V3ProverConfig::minimal().with_k(14);
        let prover = MLTrainingProverV3::with_config(arch.clone(), config);

        let w1 = vec![Fr::from(1u64); 2 * 3];
        let b1 = vec![Fr::zero(); 3];
        let w2 = vec![Fr::from(1u64); 3 * 2];
        let b2 = vec![Fr::zero(); 2];
        let w3 = vec![Fr::from(1u64); 2 * 1];
        let b3 = vec![Fr::zero(); 1];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
            (w3.as_slice(), b3.as_slice()),
        ];

        let witness = MLTrainingProverV3::build_witness(
            &arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64),
        );

        let result = prover.prove(&witness).expect("V3 prove should succeed");
        assert!(!result.proof.is_empty());
        assert!(prover.verify_result(&result));
    }

    #[test]
    fn test_v3_evm_format() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let arch = MLPArchitecture::two_layer_mlp(2, 2, 1);
        let config = V3ProverConfig::minimal();
        let prover = MLTrainingProverV3::with_config(arch.clone(), config);

        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::zero(), Fr::zero()];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::zero()];

        let x = vec![Fr::from(1u64), Fr::from(1u64)];
        let target = vec![Fr::from(5u64)];

        let layer_weights = vec![
            (w1.as_slice(), b1.as_slice()),
            (w2.as_slice(), b2.as_slice()),
        ];

        let witness = MLTrainingProverV3::build_witness(
            &arch, &x, &target, &layer_weights, Fr::from(1u64), 1, Fr::from(1u64),
        );

        let result = prover.prove(&witness).expect("prove should succeed");

        // Proof bytes must be divisible by 32 and non-empty
        assert!(!result.proof.is_empty(), "proof must not be empty");
        assert_eq!(result.proof.len() % 32, 0, "proof length must be multiple of 32");
        eprintln!("V3 proof size: {} bytes ({} x 32)", result.proof.len(), result.proof.len() / 32);

        // EVM public inputs should be 8 x 32 bytes
        let evm_pi = result.to_evm_public_inputs();
        assert_eq!(evm_pi.len(), 8);

        // Debug: verify directly through pipeline
        let pi = result.public_inputs.clone();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        let verify_result = prover.pipeline.verify(&result.proof, &pi_refs);
        eprintln!("Direct pipeline verify: {:?}", verify_result);
    }
}
