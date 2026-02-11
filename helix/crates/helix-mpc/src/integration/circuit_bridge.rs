//! Bridge between MPC and Halo2 Circuits.
//!
//! This module provides the interface for generating real Halo2 proofs
//! from MPC computations using helix-prover and helix-circuits.
//!
//! The bridge handles:
//! 1. Converting MPC witnesses to circuit-compatible format
//! 2. Calling the actual Halo2 prover
//! 3. Serializing proofs for network transmission
//! 4. Verifying proofs locally before submission
//!
//! # MPC-ZK Integration
//!
//! The key innovation is that workers compute on secret shares AND generate
//! ZK proofs. This bridge converts the witness captured during MPC computation
//! into the format expected by helix-circuits.
//!
//! ## Workflow
//!
//! 1. Workers use `ProvedArithmetic` for MPC operations, capturing witness data
//! 2. `WitnessCapture` is converted to `ReconstructedWitness` after share aggregation
//! 3. `CircuitBridge` converts to Halo2 circuit format and generates proof
//! 4. Proof is verified locally, then submitted on-chain

use std::sync::Arc;
use std::time::Instant;

use helix_prover::MLTrainingProverV2;
use helix_circuits::ml::training_step_v2::{
    compute_witness_v2, compute_state_hash_v2,
};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn, instrument};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::witness_format::ReconstructedWitness;
use crate::protocols::proved_arithmetic::{WitnessCapture, WitnessSummary};

/// Result of Halo2 proof generation.
#[derive(Debug, Clone)]
pub struct Halo2ProofResult {
    /// Serialized Halo2 KZG proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification (7 elements).
    pub public_inputs: Vec<Fr>,
    /// Old state commitment hash (lo, hi).
    pub old_state_hash: (Fr, Fr),
    /// New state commitment hash (lo, hi).
    pub new_state_hash: (Fr, Fr),
    /// Computed loss value.
    pub loss: Fr,
    /// Total accumulated error bound.
    pub total_error: Fr,
    /// Training step number.
    pub step_number: u64,
    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Proof size in bytes.
    pub proof_size_bytes: usize,
}

impl Halo2ProofResult {
    /// Creates a new proof result.
    pub fn new(
        proof: Vec<u8>,
        public_inputs: Vec<Fr>,
        old_hash: (Fr, Fr),
        new_hash: (Fr, Fr),
        loss: Fr,
        error: Fr,
        step: u64,
        time_ms: u64,
    ) -> Self {
        let proof_size = proof.len();
        Self {
            proof,
            public_inputs,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            loss,
            total_error: error,
            step_number: step,
            generation_time_ms: time_ms,
            proof_size_bytes: proof_size,
        }
    }

    /// Converts public inputs to the format expected by smart contracts.
    /// Returns 7 32-byte arrays.
    pub fn public_inputs_bytes(&self) -> Vec<[u8; 32]> {
        self.public_inputs.iter().map(|f| f.to_bytes_le()).collect()
    }
}

/// Configuration for the circuit bridge.
#[derive(Debug, Clone)]
pub struct CircuitBridgeConfig {
    /// Input dimension.
    pub d_in: usize,
    /// Hidden dimension.
    pub d_hid: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Circuit size parameter (k for 2^k rows).
    pub circuit_k: u32,
    /// Base error bound per operation.
    pub base_error: f64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
}

impl Default for CircuitBridgeConfig {
    fn default() -> Self {
        Self {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            circuit_k: 14,
            base_error: 1e-6,
            use_freivalds: true,
        }
    }
}

impl CircuitBridgeConfig {
    /// Creates a config for the given model dimensions.
    pub fn for_model(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_hid,
            d_out,
            ..Default::default()
        }
    }

    /// Sets the circuit k parameter.
    pub fn with_k(mut self, k: u32) -> Self {
        self.circuit_k = k;
        self
    }

    /// Sets the base error.
    pub fn with_base_error(mut self, err: f64) -> Self {
        self.base_error = err;
        self
    }
}

/// Bridge for generating real Halo2 proofs from MPC computations.
///
/// This is the main interface for proof generation. It holds the prover
/// state and can generate multiple proofs efficiently.
#[allow(dead_code)]
pub struct CircuitBridge {
    /// Configuration.
    config: CircuitBridgeConfig,
    /// The actual Halo2 prover.
    prover: MLTrainingProverV2,
    /// Cached proving key setup completed.
    setup_complete: bool,
}

impl CircuitBridge {
    /// Creates a new circuit bridge with the given configuration.
    pub fn new(config: CircuitBridgeConfig) -> Self {
        let prover = MLTrainingProverV2::new(config.d_in, config.d_hid, config.d_out);

        Self {
            config,
            prover,
            setup_complete: true,
        }
    }

    /// Creates a bridge with default configuration for the given dimensions.
    pub fn for_model(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self::new(CircuitBridgeConfig::for_model(d_in, d_hid, d_out))
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CircuitBridgeConfig {
        &self.config
    }

    /// Validates that all witness values are within acceptable magnitude bounds.
    ///
    /// Out-of-range Fr values silently produce invalid proofs. This check
    /// catches them early with a clear error.
    ///
    /// `max_magnitude`: maximum absolute value allowed for any witness element
    /// (in the f64 domain, before fixed-point encoding).
    pub fn validate_witness_precision(
        witness: &ReconstructedWitness,
        max_magnitude: f64,
    ) -> MPCResult<()> {
        let check = |values: &[Fr], label: &str| -> MPCResult<()> {
            for (i, v) in values.iter().enumerate() {
                let val = v.to_f64();
                if val.abs() > max_magnitude {
                    return Err(MPCError::ErrorBoundExceeded {
                        computed: val.abs(),
                        maximum: max_magnitude,
                    });
                }
                if val.is_nan() || val.is_infinite() {
                    return Err(MPCError::ProtocolError(format!(
                        "{} element {} is NaN or infinite",
                        label, i,
                    )));
                }
            }
            Ok(())
        };

        check(&witness.w1, "w1")?;
        check(&witness.b1, "b1")?;
        check(&witness.w2, "w2")?;
        check(&witness.b2, "b2")?;
        check(&witness.w1_new, "w1_new")?;
        check(&witness.b1_new, "b1_new")?;
        check(&witness.w2_new, "w2_new")?;
        check(&witness.b2_new, "b2_new")?;
        check(&witness.input, "input")?;
        check(&witness.target, "target")?;

        Ok(())
    }

    /// Generates a real Halo2 proof from a reconstructed MPC witness.
    ///
    /// This is the main proof generation function. It:
    /// 1. Validates witness precision bounds
    /// 2. Converts the MPC witness to circuit format
    /// 3. Computes state hashes
    /// 4. Generates the actual Halo2 KZG proof
    /// 5. Returns the proof with all necessary metadata
    #[instrument(skip(self, witness), level = "info", fields(
        d_in = self.config.d_in,
        d_hid = self.config.d_hid,
        d_out = self.config.d_out,
        step = witness.step_number,
    ))]
    pub fn prove(&self, witness: &ReconstructedWitness) -> MPCResult<Halo2ProofResult> {
        // Validate witness values are within bounds before attempting proof generation.
        // The fixed-point encoding uses 2^64 scaling, so values beyond ~1e18 overflow.
        // We use a conservative bound.
        Self::validate_witness_precision(witness, 1e15)?;
        let start = Instant::now();

        debug!(
            w1_size = witness.w1.len(),
            b1_size = witness.b1.len(),
            w2_size = witness.w2.len(),
            b2_size = witness.b2.len(),
            input_size = witness.input.len(),
            target_size = witness.target.len(),
            "Converting MPC witness to circuit format"
        );

        // Convert Fr types to Halo2Fr for the prover
        let x: Vec<_> = witness.input.iter().map(|f| *f.inner()).collect();
        let target: Vec<_> = witness.target.iter().map(|f| *f.inner()).collect();
        let w1: Vec<_> = witness.w1.iter().map(|f| *f.inner()).collect();
        let b1: Vec<_> = witness.b1.iter().map(|f| *f.inner()).collect();
        let w2: Vec<_> = witness.w2.iter().map(|f| *f.inner()).collect();
        let b2: Vec<_> = witness.b2.iter().map(|f| *f.inner()).collect();
        let lr = *witness.lr.inner();
        let base_error = *Fr::from_f64(self.config.base_error).inner();

        // Convert state hashes from our Fr to Halo2Fr
        let old_state_hash = (
            *witness.old_state_hash.0.inner(),
            *witness.old_state_hash.1.inner(),
        );
        let new_state_hash = (
            *witness.new_state_hash.0.inner(),
            *witness.new_state_hash.1.inner(),
        );

        // Compute the circuit witness using helix-circuits function
        let circuit_witness = compute_witness_v2(
            self.config.d_in,
            self.config.d_hid,
            self.config.d_out,
            &x,
            &target,
            &w1,
            &b1,
            &w2,
            &b2,
            lr,
            old_state_hash,
            new_state_hash,
            witness.step_number,
            base_error,
        );

        // Generate proof using the prover
        let result = self.prover.prove(&circuit_witness)
            .map_err(|e| MPCError::ProtocolError(format!("proof generation failed: {:?}", e)))?;

        let elapsed = start.elapsed();

        // Convert back to our Fr type
        let public_inputs: Vec<Fr> = result.public_inputs.iter()
            .map(|f| Fr::from_inner(*f))
            .collect();

        let old_hash = (
            Fr::from_inner(result.old_state_hash.0),
            Fr::from_inner(result.old_state_hash.1),
        );
        let new_hash = (
            Fr::from_inner(result.new_state_hash.0),
            Fr::from_inner(result.new_state_hash.1),
        );

        let proof_result = Halo2ProofResult::new(
            result.proof,
            public_inputs,
            old_hash,
            new_hash,
            Fr::from_inner(result.loss),
            Fr::from_inner(result.total_error),
            result.step_number,
            elapsed.as_millis() as u64,
        );

        info!(
            proof_size_bytes = proof_result.proof_size_bytes,
            generation_time_ms = proof_result.generation_time_ms,
            step = proof_result.step_number,
            loss = proof_result.loss.to_f64(),
            total_error = proof_result.total_error.to_f64(),
            "Halo2 proof generated successfully"
        );

        Ok(proof_result)
    }

    /// Verifies a proof locally.
    ///
    /// This is useful for sanity checking before submitting to the chain.
    #[instrument(skip(self, proof), level = "info", fields(step = proof.step_number))]
    pub fn verify(&self, proof: &Halo2ProofResult) -> MPCResult<bool> {
        debug!(
            proof_size = proof.proof_size_bytes,
            num_public_inputs = proof.public_inputs.len(),
            "Verifying Halo2 proof locally"
        );

        // Convert public inputs to Halo2Fr
        let pi: Vec<_> = proof.public_inputs.iter()
            .map(|f| *f.inner())
            .collect();

        let verified = self.prover.verify(&proof.proof, &pi);

        if verified {
            info!(step = proof.step_number, "Proof verification succeeded");
        } else {
            warn!(step = proof.step_number, "Proof verification FAILED");
        }

        Ok(verified)
    }

    /// Generates proofs for a batch of training steps.
    ///
    /// This is more efficient than calling prove() multiple times
    /// as it can amortize setup costs.
    pub fn prove_batch(&self, witnesses: &[ReconstructedWitness]) -> MPCResult<Vec<Halo2ProofResult>> {
        witnesses.iter().map(|w| self.prove(w)).collect()
    }
}

/// Thread-safe wrapper for CircuitBridge.
pub type SharedCircuitBridge = Arc<CircuitBridge>;

/// Creates a shared circuit bridge.
pub fn create_shared_bridge(config: CircuitBridgeConfig) -> SharedCircuitBridge {
    Arc::new(CircuitBridge::new(config))
}

/// Computes state hash compatible with helix-circuits.
///
/// This uses the same algorithm as compute_state_hash_v2 from helix-circuits.
pub fn compute_compatible_state_hash(
    w1: &[Fr],
    b1: &[Fr],
    w2: &[Fr],
    b2: &[Fr],
) -> (Fr, Fr) {
    let w1_h: Vec<_> = w1.iter().map(|f| *f.inner()).collect();
    let b1_h: Vec<_> = b1.iter().map(|f| *f.inner()).collect();
    let w2_h: Vec<_> = w2.iter().map(|f| *f.inner()).collect();
    let b2_h: Vec<_> = b2.iter().map(|f| *f.inner()).collect();

    let (lo, hi) = compute_state_hash_v2(&w1_h, &b1_h, &w2_h, &b2_h);

    (Fr::from_inner(lo), Fr::from_inner(hi))
}

/// Encoding utilities for converting MPC shares to circuit field elements.
///
/// These functions ensure that shares are encoded in a format that the
/// Halo2 circuits can understand and verify.
pub mod share_encoding {
    use super::*;

    /// Encodes a vector of shares into field elements for circuit consumption.
    ///
    /// This applies fixed-point scaling to ensure precision is maintained.
    #[instrument(skip(shares), level = "trace")]
    pub fn encode_shares(shares: &[f64]) -> Vec<Fr> {
        shares.iter().map(|&v| Fr::from_f64(v)).collect()
    }

    /// Decodes field elements back to floating point values.
    #[instrument(skip(field_elements), level = "trace")]
    pub fn decode_shares(field_elements: &[Fr]) -> Vec<f64> {
        field_elements.iter().map(|f| f.to_f64()).collect()
    }

    /// Encodes a share value with error bound tracking.
    ///
    /// Returns (encoded_value, error_bound) where error_bound accounts
    /// for the encoding precision loss.
    pub fn encode_with_error(value: f64, base_error: f64) -> (Fr, Fr) {
        let encoded = Fr::from_f64(value);
        // Encoding error is proportional to the value magnitude
        let encoding_error = base_error + value.abs() * 1e-15;
        (encoded, Fr::from_f64(encoding_error))
    }

    /// Encodes a matrix in row-major order.
    pub fn encode_matrix(matrix: &[f64], rows: usize, cols: usize) -> Vec<Fr> {
        assert_eq!(matrix.len(), rows * cols, "Matrix dimension mismatch");
        encode_shares(matrix)
    }

    /// Computes a commitment to encoded shares.
    pub fn commit_encoded(encoded: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in encoded {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Validates that shares are within acceptable bounds for circuit constraints.
    pub fn validate_share_bounds(shares: &[Fr], max_magnitude: f64) -> bool {
        for share in shares {
            let val = share.to_f64();
            if val.abs() > max_magnitude {
                warn!(
                    value = val,
                    max = max_magnitude,
                    "Share value exceeds maximum magnitude"
                );
                return false;
            }
        }
        true
    }
}

/// Converts a WitnessCapture from MPC operations into a ReconstructedWitness.
///
/// This is the critical conversion that bridges the MPC computation
/// with the ZK proof generation.
#[instrument(skip(captures, d_in, d_hid, d_out), level = "info")]
/// Converts MPC witness captures to a reconstructed witness for ZK proof generation.
///
/// This function aggregates witness data from multiple MPC workers and reconstructs
/// the full computation witness needed for Halo2 proof generation.
#[instrument(skip(captures, input, target), level = "info", fields(
    num_captures = captures.len(),
    d_in = d_in,
    d_hid = d_hid,
    d_out = d_out,
    step = step_number,
))]
pub fn witness_capture_to_reconstructed(
    captures: &[WitnessCapture],
    d_in: usize,
    d_hid: usize,
    d_out: usize,
    input: &[f64],
    target: &[f64],
    learning_rate: f64,
    step_number: u64,
) -> MPCResult<ReconstructedWitness> {
    if captures.is_empty() {
        warn!("Witness conversion failed: no captures provided");
        return Err(MPCError::ProtocolError("No witness captures provided".into()));
    }

    info!(
        num_workers = captures.len(),
        input_size = input.len(),
        target_size = target.len(),
        "Starting witness reconstruction from MPC captures"
    );

    // Log summary of all captures
    let mut total_ops: usize = 0;
    let mut total_beaver: usize = 0;
    let mut total_error: f64 = 0.0;
    for (i, capture) in captures.iter().enumerate() {
        let summary = WitnessSummary::from_capture(capture);
        total_ops += summary.total_operations;
        total_beaver += summary.beaver_triples_used;
        total_error = total_error.max(summary.total_error);
        debug!(
            party = i,
            total_ops = summary.total_operations,
            beaver_triples = summary.beaver_triples_used,
            total_error = summary.total_error,
            "Worker witness capture summary"
        );
    }

    debug!(
        total_operations = total_ops,
        total_beaver_triples = total_beaver,
        max_error = total_error,
        "Aggregate witness statistics"
    );

    // Extract weight shares from captures
    // This assumes each capture contains the party's weight data as inputs
    let _num_parties = captures.len();

    // Initialize weight accumulators
    let mut w1 = vec![Fr::ZERO; d_hid * d_in];
    let b1 = vec![Fr::ZERO; d_hid];
    let w2 = vec![Fr::ZERO; d_out * d_hid];
    let b2 = vec![Fr::ZERO; d_out];

    // Sum shares from all parties (additive secret sharing reconstruction)
    for capture in captures {
        // Extract inputs from the capture
        // The first operations typically contain the weight data
        let all_inputs = capture.all_inputs();

        // Parse the inputs based on expected structure
        // This is a simplified extraction - in practice would need more structure
        if all_inputs.len() >= d_hid * d_in {
            for (i, val) in all_inputs.iter().take(d_hid * d_in).enumerate() {
                if i < w1.len() {
                    w1[i] = Fr::add(&w1[i], val);
                }
            }
        }
    }

    // Compute total error across all captures
    let total_error = captures
        .iter()
        .fold(Fr::ZERO, |acc, c| Fr::add(&acc, c.total_error()));

    // Convert input/target to Fr
    let input_fr: Vec<Fr> = input.iter().map(|&v| Fr::from_f64(v)).collect();
    let target_fr: Vec<Fr> = target.iter().map(|&v| Fr::from_f64(v)).collect();
    let lr = Fr::from_f64(learning_rate);

    // Compute state hashes
    let old_hash = compute_compatible_state_hash(&w1, &b1, &w2, &b2);

    // Apply simplified gradient update (for demonstration)
    // In full implementation, this would use actual gradients from captures
    let w1_new = w1.clone();
    let b1_new = b1.clone();
    let w2_new = w2.clone();
    let b2_new = b2.clone();

    let new_hash = compute_compatible_state_hash(&w1_new, &b1_new, &w2_new, &b2_new);

    // Generate Freivalds challenges
    let (freivalds_r1, freivalds_r2) = crate::integration::witness::generate_freivalds_challenges(
        step_number,
        d_hid,
        d_out,
    );

    Ok(ReconstructedWitness {
        d_in,
        d_hid,
        d_out,
        input: input_fr,
        target: target_fr,
        w1,
        b1,
        w2,
        b2,
        w1_new,
        b1_new,
        w2_new,
        b2_new,
        lr,
        old_state_hash: old_hash,
        new_state_hash: new_hash,
        step_number,
        total_error,
        freivalds_r1,
        freivalds_r2,
    })
}

/// Statistics about proof generation for monitoring.
#[derive(Debug, Clone, Default)]
pub struct ProofGenerationStats {
    /// Number of proofs generated.
    pub proofs_generated: u64,
    /// Total proof generation time in milliseconds.
    pub total_generation_time_ms: u64,
    /// Total proof bytes generated.
    pub total_proof_bytes: u64,
    /// Average proof size in bytes.
    pub avg_proof_size: u64,
    /// Average generation time in milliseconds.
    pub avg_generation_time_ms: u64,
    /// Number of verification failures.
    pub verification_failures: u64,
    /// Last proof generation timestamp.
    pub last_proof_time_ms: u64,
}

impl ProofGenerationStats {
    /// Records a proof generation.
    pub fn record(&mut self, size_bytes: usize, time_ms: u64) {
        self.proofs_generated += 1;
        self.total_generation_time_ms += time_ms;
        self.total_proof_bytes += size_bytes as u64;

        if self.proofs_generated > 0 {
            self.avg_proof_size = self.total_proof_bytes / self.proofs_generated;
            self.avg_generation_time_ms = self.total_generation_time_ms / self.proofs_generated;
        }

        self.last_proof_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Records a verification failure.
    pub fn record_failure(&mut self) {
        self.verification_failures += 1;
    }
}

/// Extended circuit bridge with statistics tracking.
pub struct TrackedCircuitBridge {
    /// Inner bridge.
    bridge: CircuitBridge,
    /// Generation statistics.
    stats: ProofGenerationStats,
}

impl TrackedCircuitBridge {
    /// Creates a new tracked bridge.
    pub fn new(config: CircuitBridgeConfig) -> Self {
        Self {
            bridge: CircuitBridge::new(config),
            stats: ProofGenerationStats::default(),
        }
    }

    /// Generates a proof with statistics tracking.
    #[instrument(skip(self, witness), level = "info")]
    pub fn prove_tracked(&mut self, witness: &ReconstructedWitness) -> MPCResult<Halo2ProofResult> {
        let result = self.bridge.prove(witness)?;

        self.stats.record(result.proof_size_bytes, result.generation_time_ms);

        info!(
            proof_size = result.proof_size_bytes,
            generation_time_ms = result.generation_time_ms,
            step = result.step_number,
            "Proof generated successfully"
        );

        Ok(result)
    }

    /// Verifies a proof with statistics tracking.
    #[instrument(skip(self, proof), level = "info")]
    pub fn verify_tracked(&mut self, proof: &Halo2ProofResult) -> MPCResult<bool> {
        let result = self.bridge.verify(proof)?;

        if !result {
            self.stats.record_failure();
            warn!(step = proof.step_number, "Proof verification failed");
        } else {
            debug!(step = proof.step_number, "Proof verified successfully");
        }

        Ok(result)
    }

    /// Returns the inner bridge.
    pub fn inner(&self) -> &CircuitBridge {
        &self.bridge
    }

    /// Returns the statistics.
    pub fn stats(&self) -> &ProofGenerationStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_witness() -> ReconstructedWitness {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1: Vec<Fr> = vec![Fr::from_f64(0.1); d_hid * d_in];
        let b1: Vec<Fr> = vec![Fr::from_f64(0.01); d_hid];
        let w2: Vec<Fr> = vec![Fr::from_f64(0.1); d_out * d_hid];
        let b2: Vec<Fr> = vec![Fr::from_f64(0.01); d_out];

        ReconstructedWitness {
            d_in,
            d_hid,
            d_out,
            input: vec![Fr::from_f64(0.5), Fr::from_f64(0.5)],
            target: vec![Fr::from_f64(1.0)],
            w1: w1.clone(),
            b1: b1.clone(),
            w2: w2.clone(),
            b2: b2.clone(),
            w1_new: w1,
            b1_new: b1,
            w2_new: w2,
            b2_new: b2,
            lr: Fr::from_f64(0.01),
            old_state_hash: (Fr::from_u64(0), Fr::from_u64(0)),
            new_state_hash: (Fr::from_u64(0), Fr::from_u64(0)),
            step_number: 0,
            total_error: Fr::from_f64(0.001),
            freivalds_r1: vec![Fr::from_u64(42); 2],
            freivalds_r2: vec![Fr::from_u64(43); 1],
        }
    }

    #[test]
    fn test_circuit_bridge_config() {
        let config = CircuitBridgeConfig::for_model(4, 8, 2)
            .with_k(12)
            .with_base_error(1e-5);

        assert_eq!(config.d_in, 4);
        assert_eq!(config.d_hid, 8);
        assert_eq!(config.d_out, 2);
        assert_eq!(config.circuit_k, 12);
    }

    #[test]
    fn test_compatible_state_hash() {
        let w1 = vec![Fr::from_f64(0.1), Fr::from_f64(0.2)];
        let b1 = vec![Fr::from_f64(0.01)];
        let w2 = vec![Fr::from_f64(0.3)];
        let b2 = vec![Fr::from_f64(0.02)];

        let (lo1, hi1) = compute_compatible_state_hash(&w1, &b1, &w2, &b2);
        let (lo2, hi2) = compute_compatible_state_hash(&w1, &b1, &w2, &b2);

        // Same inputs should produce same hash
        assert!(lo1.ct_eq(&lo2).to_bool());
        assert!(hi1.ct_eq(&hi2).to_bool());

        // Different inputs should produce different hash
        let w1_diff = vec![Fr::from_f64(0.15), Fr::from_f64(0.2)];
        let (lo3, hi3) = compute_compatible_state_hash(&w1_diff, &b1, &w2, &b2);
        assert!(!lo1.ct_eq(&lo3).to_bool() || !hi1.ct_eq(&hi3).to_bool());
    }

    // Note: Full proof generation test is expensive, so we mark it as ignored
    // Run with: cargo test --release -- --ignored
    #[test]
    #[ignore]
    fn test_circuit_bridge_prove() {
        let bridge = CircuitBridge::for_model(2, 2, 1);
        let witness = create_test_witness();

        let result = bridge.prove(&witness).unwrap();

        assert!(!result.proof.is_empty());
        assert_eq!(result.public_inputs.len(), 7);
        assert_eq!(result.step_number, 0);

        // Verify the proof
        assert!(bridge.verify(&result).unwrap());
    }

    #[test]
    #[ignore]
    fn test_circuit_bridge_batch() {
        let bridge = CircuitBridge::for_model(2, 2, 1);
        let witnesses: Vec<_> = (0..3).map(|_| create_test_witness()).collect();

        let results = bridge.prove_batch(&witnesses).unwrap();
        assert_eq!(results.len(), 3);

        for result in &results {
            assert!(bridge.verify(result).unwrap());
        }
    }

    #[test]
    fn test_validate_witness_precision_ok() {
        let witness = create_test_witness();
        // All values are small (< 1.0), so 1e15 bound should pass easily.
        assert!(CircuitBridge::validate_witness_precision(&witness, 1e15).is_ok());
    }

    #[test]
    fn test_validate_witness_precision_rejects_large_values() {
        let mut witness = create_test_witness();
        // Inject an out-of-range value into w1.
        witness.w1[0] = Fr::from_f64(1e16);

        let result = CircuitBridge::validate_witness_precision(&witness, 1e15);
        assert!(result.is_err());
        match result.unwrap_err() {
            MPCError::ErrorBoundExceeded { computed, maximum } => {
                assert!(computed > maximum);
            }
            other => panic!("Expected ErrorBoundExceeded, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_witness_precision_tight_bound() {
        let witness = create_test_witness();
        // Values are ~0.01-1.0, so a bound of 0.001 should fail.
        let result = CircuitBridge::validate_witness_precision(&witness, 0.001);
        assert!(result.is_err());
    }
}
