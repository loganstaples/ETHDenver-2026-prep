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

use std::sync::Arc;
use std::time::Instant;

use helix_prover::MLTrainingProverV2;
use helix_circuits::ml::training_step_v2::{
    MLTrainingStepV2Witness, MLTrainingStepV2Circuit,
    compute_witness_v2, compute_state_hash_v2,
};

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::witness::CircuitWitness;
use crate::integration::witness_format::ReconstructedWitness;

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

    /// Generates a real Halo2 proof from a reconstructed MPC witness.
    ///
    /// This is the main proof generation function. It:
    /// 1. Converts the MPC witness to circuit format
    /// 2. Computes state hashes
    /// 3. Generates the actual Halo2 KZG proof
    /// 4. Returns the proof with all necessary metadata
    pub fn prove(&self, witness: &ReconstructedWitness) -> MPCResult<Halo2ProofResult> {
        let start = Instant::now();

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
        let result = self.prover.prove(&circuit_witness);

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

        Ok(Halo2ProofResult::new(
            result.proof,
            public_inputs,
            old_hash,
            new_hash,
            Fr::from_inner(result.loss),
            Fr::from_inner(result.total_error),
            result.step_number,
            elapsed.as_millis() as u64,
        ))
    }

    /// Verifies a proof locally.
    ///
    /// This is useful for sanity checking before submitting to the chain.
    pub fn verify(&self, proof: &Halo2ProofResult) -> MPCResult<bool> {
        // Convert public inputs to Halo2Fr
        let pi: Vec<_> = proof.public_inputs.iter()
            .map(|f| *f.inner())
            .collect();

        let verified = self.prover.verify(&proof.proof, &pi);

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
}
