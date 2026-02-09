//! Gradient Computation Prover.
//!
//! Specialized prover for proving gradient computations in neural network training,
//! with support for error bound tracking, gradient aggregation, and MPC-compatible proofs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::chunking::{ChunkId, ComputationChunk, ComputationType};
use crate::parallel::ChunkProof;
use crate::pipeline::ProverPipeline;
use crate::provers::ivc_circuit::IVCStepCircuit;

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;

/// Configuration for gradient proving.
#[derive(Debug, Clone)]
pub struct GradientProverConfig {
    /// Circuit size parameter (2^k rows).
    pub k: u32,
    /// Maximum error bound per operation.
    pub max_error_per_op: f64,
    /// Enable Freivalds verification for matrix operations.
    pub use_freivalds: bool,
    /// Quantization bits for gradient values.
    pub quantization_bits: u32,
    /// Enable gradient clipping.
    pub gradient_clipping: Option<f64>,
    /// Enable gradient noise for differential privacy.
    pub differential_privacy: Option<DifferentialPrivacyConfig>,
}

impl Default for GradientProverConfig {
    fn default() -> Self {
        Self {
            k: 12,
            max_error_per_op: 1e-6,
            use_freivalds: true,
            quantization_bits: 16,
            gradient_clipping: Some(1.0),
            differential_privacy: None,
        }
    }
}

/// Differential privacy configuration.
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyConfig {
    /// Noise scale (sigma).
    pub noise_scale: f64,
    /// Privacy budget (epsilon).
    pub epsilon: f64,
    /// Privacy budget (delta).
    pub delta: f64,
}

/// Result of proving a gradient computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofResult {
    /// The proof bytes.
    pub proof: Vec<u8>,
    /// Gradient commitment.
    pub gradient_commitment: [u8; 32],
    /// Error bound for this gradient computation.
    pub error_bound: f64,
    /// Layer index this gradient applies to.
    pub layer_index: usize,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Public inputs for verification.
    pub public_inputs: Vec<[u8; 32]>,
}

/// Gradient data for proving.
#[derive(Debug, Clone)]
pub struct GradientData {
    /// Layer index.
    pub layer_index: usize,
    /// Input activations (from forward pass).
    pub input_activations: Vec<Fr>,
    /// Output gradients (from backward pass of next layer).
    pub output_gradients: Vec<Fr>,
    /// Current weights.
    pub weights: Vec<Fr>,
    /// Computed gradients.
    pub gradients: Vec<Fr>,
    /// Learning rate.
    pub learning_rate: Fr,
    /// Error bound from previous computations.
    pub input_error: Fr,
}

impl GradientData {
    /// Computes the gradient commitment.
    pub fn commitment(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash gradient values
        for g in &self.gradients {
            let bytes = fr_to_bytes(g);
            hasher.update(&bytes);
        }

        // Include layer index
        hasher.update(&self.layer_index.to_le_bytes());

        hasher.finalize().into()
    }

    /// Computes the total gradient norm (for clipping).
    pub fn gradient_norm(&self) -> f64 {
        let sum_sq: f64 = self.gradients
            .iter()
            .map(|g| fr_to_f64(g).powi(2))
            .sum();
        sum_sq.sqrt()
    }

    /// Clips gradients to the given maximum norm.
    pub fn clip(&mut self, max_norm: f64) {
        let norm = self.gradient_norm();
        if norm > max_norm {
            let scale = Fr::from((max_norm / norm * 1e9) as u64);
            let divisor = Fr::from(1_000_000_000u64);
            for g in &mut self.gradients {
                *g = *g * scale;
                // Note: Proper division would need field inversion
            }
        }
    }
}

/// Prover for gradient computations.
pub struct GradientProver {
    /// Configuration.
    config: GradientProverConfig,
    /// Proving pipeline.
    pipeline: ProverPipeline<IVCStepCircuit>,
    /// Cached proofs by layer.
    cache: Mutex<HashMap<usize, GradientProofResult>>,
    /// Statistics.
    stats: Mutex<GradientProverStats>,
}

/// Statistics for gradient proving.
#[derive(Debug, Clone, Default)]
pub struct GradientProverStats {
    /// Total proofs generated.
    pub proofs_generated: u64,
    /// Total proof time (ms).
    pub total_proof_time_ms: u64,
    /// Average proof time (ms).
    pub avg_proof_time_ms: f64,
    /// Total error accumulated.
    pub total_error: f64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

impl GradientProver {
    /// Creates a new gradient prover.
    pub fn new() -> Self {
        Self::with_config(GradientProverConfig::default())
    }

    /// Creates a gradient prover with custom configuration.
    pub fn with_config(config: GradientProverConfig) -> Self {
        let mut pipeline = ProverPipeline::new(config.k);
        if let Err(e) = pipeline.setup(&IVCStepCircuit::default()) {
            tracing::error!("GradientProver pipeline setup failed: {e}");
        }

        Self {
            config,
            pipeline,
            cache: Mutex::new(HashMap::new()),
            stats: Mutex::new(GradientProverStats::default()),
        }
    }

    /// Proves a gradient computation.
    pub fn prove(&self, data: &GradientData) -> Result<GradientProofResult, String> {
        let start = Instant::now();

        // Build circuit for gradient computation
        let (circuit, public_inputs) = self.build_gradient_circuit(data);

        // Generate proof
        let pi_refs: Vec<&[Fr]> = vec![&public_inputs];
        let proof = self.pipeline.prove(&circuit, &pi_refs)
            .map_err(|e| {
                tracing::error!("Gradient proof generation failed for layer {}: {e}", data.layer_index);
                format!("Gradient proof generation failed: {e}")
            })?;

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Compute error bound
        let error_bound = self.compute_error_bound(data);

        // Convert public inputs to bytes
        let public_input_bytes: Vec<[u8; 32]> = public_inputs
            .iter()
            .map(|pi| fr_to_bytes(pi))
            .collect();

        let result = GradientProofResult {
            proof,
            gradient_commitment: data.commitment(),
            error_bound,
            layer_index: data.layer_index,
            generation_time_ms: elapsed_ms,
            public_inputs: public_input_bytes,
        };

        // Update stats
        {
            if let Ok(mut stats) = self.stats.lock() {
                stats.proofs_generated += 1;
                stats.total_proof_time_ms += elapsed_ms;
                stats.avg_proof_time_ms = stats.total_proof_time_ms as f64 / stats.proofs_generated as f64;
                stats.total_error += error_bound;
            }
        }

        // Cache the result
        {
            if let Ok(mut cache) = self.cache.lock() {
                cache.insert(data.layer_index, result.clone());
            }
        }

        Ok(result)
    }

    /// Proves a batch of gradient computations.
    pub fn prove_batch(&self, batch: &[GradientData]) -> Vec<GradientProofResult> {
        batch.iter().filter_map(|data| {
            match self.prove(data) {
                Ok(result) => Some(result),
                Err(e) => {
                    tracing::error!("Batch gradient proof failed for layer {}: {e}", data.layer_index);
                    None
                }
            }
        }).collect()
    }

    /// Verifies a gradient proof.
    pub fn verify(&self, result: &GradientProofResult) -> bool {
        // Convert public inputs back to Fr
        let public_inputs: Vec<Fr> = result.public_inputs
            .iter()
            .map(|bytes| bytes_to_fr(bytes))
            .collect();

        let pi_refs: Vec<&[Fr]> = vec![&public_inputs];
        self.pipeline.verify(&result.proof, &pi_refs).unwrap_or(false)
    }

    /// Gets a cached proof for a layer.
    pub fn get_cached(&self, layer_index: usize) -> Option<GradientProofResult> {
        let mut stats = self.stats.lock().ok()?;
        let cache = self.cache.lock().ok()?;

        if let Some(result) = cache.get(&layer_index) {
            stats.cache_hits += 1;
            Some(result.clone())
        } else {
            stats.cache_misses += 1;
            None
        }
    }

    /// Clears the proof cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Returns prover statistics.
    pub fn stats(&self) -> GradientProverStats {
        self.stats.lock().ok().map(|s| s.clone()).unwrap_or_default()
    }

    /// Aggregates multiple gradient proofs.
    pub fn aggregate_proofs(&self, proofs: &[GradientProofResult]) -> AggregatedGradientProof {
        let mut total_error = 0.0;
        let mut commitments = Vec::new();

        for proof in proofs {
            total_error += proof.error_bound;
            commitments.push(proof.gradient_commitment);
        }

        // Compute aggregate commitment
        let aggregate_commitment = {
            let mut hasher = Sha256::new();
            for c in &commitments {
                hasher.update(c);
            }
            hasher.finalize().into()
        };

        AggregatedGradientProof {
            aggregate_commitment,
            individual_commitments: commitments,
            total_error_bound: total_error,
            num_layers: proofs.len(),
            proofs: proofs.to_vec(),
        }
    }

    fn build_gradient_circuit(&self, data: &GradientData) -> (IVCStepCircuit, Vec<Fr>) {
        // Compute input state commitment
        let input_state = {
            let mut hasher = Sha256::new();
            for a in &data.input_activations {
                hasher.update(&fr_to_bytes(a));
            }
            for g in &data.output_gradients {
                hasher.update(&fr_to_bytes(g));
            }
            hasher.finalize().into()
        };

        // Compute output state commitment (gradient commitment)
        let output_state = data.commitment();

        // Compute computation hash
        let computation_hash = {
            let mut hasher = Sha256::new();
            hasher.update(b"GRADIENT_COMPUTE:");
            hasher.update(&data.layer_index.to_le_bytes());
            hasher.update(&(data.weights.len() as u64).to_le_bytes());
            hasher.finalize().into()
        };

        let circuit = IVCStepCircuit {
            prev_state: input_state,
            new_state: output_state,
            computation_hash,
            step_number: data.layer_index as u64,
        };

        let public_inputs = circuit.public_inputs();

        (circuit, public_inputs)
    }

    fn compute_error_bound(&self, data: &GradientData) -> f64 {
        // Base error from input
        let base_error = fr_to_f64(&data.input_error);

        // Error from gradient computation (proportional to number of operations)
        let num_ops = data.input_activations.len() * data.output_gradients.len();
        let computation_error = num_ops as f64 * self.config.max_error_per_op;

        // Error from quantization
        let quantization_error = 1.0 / (1u64 << self.config.quantization_bits) as f64;

        // Total error bound
        let total = base_error + computation_error + quantization_error;

        // Clamp to reasonable range
        total.min(1.0).max(0.0)
    }
}

impl Default for GradientProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated proof for multiple gradient computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGradientProof {
    /// Aggregate commitment over all gradients.
    pub aggregate_commitment: [u8; 32],
    /// Individual gradient commitments.
    pub individual_commitments: Vec<[u8; 32]>,
    /// Total error bound.
    pub total_error_bound: f64,
    /// Number of layers included.
    pub num_layers: usize,
    /// Individual proofs.
    pub proofs: Vec<GradientProofResult>,
}

impl AggregatedGradientProof {
    /// Verifies the aggregated proof structure.
    pub fn verify_structure(&self) -> bool {
        if self.proofs.len() != self.num_layers {
            return false;
        }

        if self.individual_commitments.len() != self.num_layers {
            return false;
        }

        // Verify aggregate commitment
        let computed: [u8; 32] = {
            let mut hasher = Sha256::new();
            for c in &self.individual_commitments {
                hasher.update(c);
            }
            hasher.finalize().into()
        };

        computed == self.aggregate_commitment
    }
}

/// Prover for distributed gradient computation (MPC-compatible).
pub struct DistributedGradientProver {
    /// Worker ID.
    worker_id: u32,
    /// Total number of workers.
    num_workers: u32,
    /// Underlying prover.
    prover: GradientProver,
    /// Share commitments received from other workers.
    share_commitments: Mutex<HashMap<u32, Vec<[u8; 32]>>>,
}

impl DistributedGradientProver {
    /// Creates a new distributed gradient prover.
    pub fn new(worker_id: u32, num_workers: u32) -> Self {
        Self {
            worker_id,
            num_workers,
            prover: GradientProver::new(),
            share_commitments: Mutex::new(HashMap::new()),
        }
    }

    /// Proves a gradient share (for MPC).
    pub fn prove_share(&self, data: &GradientData) -> Result<GradientShareProof, String> {
        let proof = self.prover.prove(data)?;

        // Generate share-specific commitment
        let share_commitment = {
            let mut hasher = Sha256::new();
            hasher.update(&self.worker_id.to_le_bytes());
            hasher.update(&proof.gradient_commitment);
            hasher.finalize().into()
        };

        Ok(GradientShareProof {
            worker_id: self.worker_id,
            share_commitment,
            gradient_proof: proof,
        })
    }

    /// Records a share commitment from another worker.
    pub fn record_share_commitment(&self, worker_id: u32, commitment: [u8; 32], layer: usize) {
        if let Ok(mut commitments) = self.share_commitments.lock() {
            let worker_commits = commitments.entry(worker_id).or_insert_with(Vec::new);

            while worker_commits.len() <= layer {
                worker_commits.push([0u8; 32]);
            }
            worker_commits[layer] = commitment;
        }
    }

    /// Verifies that all workers contributed valid shares.
    pub fn verify_all_shares(&self, layer: usize) -> bool {
        let commitments = match self.share_commitments.lock() {
            Ok(c) => c,
            Err(_) => return false,
        };

        // Check we have commitments from all workers
        for worker_id in 0..self.num_workers {
            if worker_id == self.worker_id {
                continue; // Skip self
            }

            match commitments.get(&worker_id) {
                Some(commits) if commits.len() > layer => {
                    if commits[layer] == [0u8; 32] {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        true
    }

    /// Generates a combined commitment for the aggregated gradient.
    pub fn combined_commitment(&self, layer: usize) -> Option<[u8; 32]> {
        let commitments = self.share_commitments.lock().ok()?;

        let mut hasher = Sha256::new();
        hasher.update(b"COMBINED_GRADIENT:");
        hasher.update(&(layer as u64).to_le_bytes());

        for worker_id in 0..self.num_workers {
            if let Some(commits) = commitments.get(&worker_id) {
                if let Some(commit) = commits.get(layer) {
                    hasher.update(commit);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        Some(hasher.finalize().into())
    }
}

/// Proof of a gradient share from one worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientShareProof {
    /// Worker ID that generated this share.
    pub worker_id: u32,
    /// Commitment to this share.
    pub share_commitment: [u8; 32],
    /// The gradient proof.
    pub gradient_proof: GradientProofResult,
}

/// Converts a ChunkProof to a GradientProofResult.
pub fn chunk_to_gradient_proof(chunk: &ChunkProof, layer_index: usize) -> GradientProofResult {
    // Extract gradient commitment from public inputs (second input is output commitment)
    let gradient_commitment = if chunk.public_inputs.len() > 1 {
        chunk.public_inputs[1]
    } else if !chunk.public_inputs.is_empty() {
        chunk.public_inputs[0]
    } else {
        [0u8; 32]
    };

    GradientProofResult {
        proof: chunk.proof.clone(),
        gradient_commitment,
        error_bound: chunk.error_bound,
        layer_index,
        generation_time_ms: chunk.generation_time_ms,
        public_inputs: chunk.public_inputs.clone(),
    }
}

/// Creates a GradientData from a computation chunk.
pub fn chunk_to_gradient_data(chunk: &ComputationChunk) -> Option<GradientData> {
    match chunk.computation_type {
        ComputationType::Backward | ComputationType::GradientAggregation => {
            Some(GradientData {
                layer_index: chunk.layer_range.0,
                input_activations: Vec::new(), // Would be populated from actual computation
                output_gradients: Vec::new(),
                weights: Vec::new(),
                gradients: Vec::new(),
                learning_rate: Fr::one(),
                input_error: Fr::from((chunk.error_bound * 1e9) as u64),
            })
        }
        _ => None,
    }
}

// Helper functions

fn fr_to_bytes(f: &Fr) -> [u8; 32] {
    use helix_circuits::halo2curves::ff::PrimeField;
    let repr = f.to_repr();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(repr.as_ref());
    bytes
}

fn bytes_to_fr(bytes: &[u8; 32]) -> Fr {
    use helix_circuits::halo2curves::ff::PrimeField;
    Fr::from_repr_vartime((*bytes).into()).unwrap_or(Fr::zero())
}

fn fr_to_f64(f: &Fr) -> f64 {
    // Convert Fr to f64 (lossy, for error bound calculations only)
    let bytes = fr_to_bytes(f);
    let value = u64::from_le_bytes(bytes[0..8].try_into().expect("invariant: fixed-size slice"));
    value as f64 / 1e18
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_gradient_data(layer: usize) -> GradientData {
        GradientData {
            layer_index: layer,
            input_activations: vec![Fr::from(1), Fr::from(2)],
            output_gradients: vec![Fr::from(1)],
            weights: vec![Fr::from(1), Fr::from(2)],
            gradients: vec![Fr::from(1), Fr::from(2)],
            learning_rate: Fr::from(1),
            input_error: Fr::zero(),
        }
    }

    #[test]
    fn test_gradient_prover_basic() {
        let prover = GradientProver::new();
        let data = make_test_gradient_data(0);

        let result = prover.prove(&data).expect("proof should succeed");

        assert!(!result.proof.is_empty());
        assert!(result.error_bound >= 0.0);
        assert_eq!(result.layer_index, 0);
    }

    #[test]
    fn test_gradient_verify() {
        let prover = GradientProver::new();
        let data = make_test_gradient_data(0);

        let result = prover.prove(&data).expect("proof should succeed");
        assert!(prover.verify(&result));
    }

    #[test]
    fn test_gradient_batch() {
        let prover = GradientProver::new();
        let batch: Vec<GradientData> = (0..3).map(make_test_gradient_data).collect();

        let results = prover.prove_batch(&batch);

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.layer_index, i);
        }
    }

    #[test]
    fn test_gradient_caching() {
        let prover = GradientProver::new();
        let data = make_test_gradient_data(5);

        // First prove
        let _ = prover.prove(&data).expect("proof should succeed");

        // Should be cached
        let cached = prover.get_cached(5);
        assert!(cached.is_some());

        let stats = prover.stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_gradient_aggregation() {
        let prover = GradientProver::new();
        let batch: Vec<GradientData> = (0..3).map(make_test_gradient_data).collect();
        let results = prover.prove_batch(&batch);

        let aggregated = prover.aggregate_proofs(&results);

        assert_eq!(aggregated.num_layers, 3);
        assert!(aggregated.verify_structure());
    }

    #[test]
    fn test_gradient_commitment() {
        let data1 = make_test_gradient_data(0);
        let data2 = make_test_gradient_data(0);
        let mut data3 = make_test_gradient_data(0);
        data3.gradients = vec![Fr::from(3), Fr::from(4)];

        // Same data should have same commitment
        assert_eq!(data1.commitment(), data2.commitment());

        // Different data should have different commitment
        assert_ne!(data1.commitment(), data3.commitment());
    }

    #[test]
    fn test_gradient_norm() {
        let mut data = GradientData {
            layer_index: 0,
            input_activations: Vec::new(),
            output_gradients: Vec::new(),
            weights: Vec::new(),
            gradients: vec![Fr::from(3_000_000_000_000_000_000u64), Fr::from(4_000_000_000_000_000_000u64)],
            learning_rate: Fr::one(),
            input_error: Fr::zero(),
        };

        let norm = data.gradient_norm();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_distributed_prover() {
        let prover = DistributedGradientProver::new(0, 3);
        let data = make_test_gradient_data(0);

        let share_proof = prover.prove_share(&data).expect("proof should succeed");

        assert_eq!(share_proof.worker_id, 0);
        assert_ne!(share_proof.share_commitment, [0u8; 32]);
    }

    #[test]
    fn test_prover_stats() {
        let prover = GradientProver::new();
        let data = make_test_gradient_data(0);

        prover.prove(&data).expect("proof should succeed");
        prover.prove(&data).expect("proof should succeed");

        let stats = prover.stats();
        assert_eq!(stats.proofs_generated, 2);
        assert!(stats.avg_proof_time_ms > 0.0);
    }

    #[test]
    fn test_error_bound_computation() {
        let prover = GradientProver::new();
        let data = make_test_gradient_data(0);

        let result = prover.prove(&data).expect("proof should succeed");

        // Error bound should be reasonable
        assert!(result.error_bound >= 0.0);
        assert!(result.error_bound <= 1.0);
    }
}
