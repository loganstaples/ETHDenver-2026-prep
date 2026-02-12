//! RLC Aggregation Prover.
//!
//! Wraps the `SHPLONKAggregationCircuit` from `helix-circuits` with a
//! `ProverPipeline` to produce real KZG proofs for aggregated training batches.
//!
//! ## Usage
//!
//! ```ignore
//! use helix_prover::aggregation::rlc_aggregation::RLCAggregationProver;
//!
//! let prover = RLCAggregationProver::new(16, 14);
//! let agg = prover.aggregate(&training_proofs)?;
//! let valid = prover.verify(&agg)?;
//! ```

use std::time::{Duration, Instant};

use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::{
    SHPLONKAggregationCircuit, SHPLONKAggregationWitness,
    AggregationStepWitness, step_witness_from_public_inputs,
    AGGREGATION_NUM_PUBLIC_INPUTS, MAX_AGGREGATION_BATCH,
};

use crate::keys::generate_keys_for_circuit;
use crate::pipeline::ProverPipeline;
use crate::provers::training_prover_v2::TrainingProofResultV2;

/// Result of RLC aggregation — single proof for N training steps.
#[derive(Debug, Clone)]
pub struct AggregatedTrainingProof {
    /// Single KZG proof bytes.
    pub proof: Vec<u8>,
    /// 8 public inputs (matches contract interface).
    pub public_inputs: Vec<Fr>,
    /// Number of individual proofs aggregated.
    pub num_steps: usize,
    /// Total accumulated loss across all steps.
    pub total_loss: Fr,
    /// Total accumulated error bound across all steps.
    pub total_error: Fr,
    /// RLC commitment (Fiat-Shamir bound).
    pub rlc_commitment: Fr,
    /// First step's old state hash (lo, hi).
    pub first_old_hash: (Fr, Fr),
    /// Last step's new state hash (lo, hi).
    pub last_new_hash: (Fr, Fr),
    /// Individual proof results (kept for reference).
    pub individual_proofs: Vec<TrainingProofResultV2>,
    /// Total aggregation time.
    pub aggregation_time: Duration,
}

impl AggregatedTrainingProof {
    /// Public inputs as byte arrays for EVM.
    pub fn to_evm_public_inputs(&self) -> Vec<[u8; 32]> {
        use helix_circuits::verifier::fr_to_evm_bytes;
        self.public_inputs
            .iter()
            .map(|fr| fr_to_evm_bytes(fr))
            .collect()
    }
}

/// RLC Aggregation Prover.
///
/// Produces a single aggregated proof for N training step proofs using
/// the `SHPLONKAggregationCircuit` which enforces PI chaining, Fiat-Shamir
/// RLC commitment, and error accumulation.
pub struct RLCAggregationProver {
    /// Proving pipeline for the aggregation circuit.
    pipeline: ProverPipeline<SHPLONKAggregationCircuit>,
    /// Maximum batch size.
    max_batch_size: usize,
    /// Whether the pipeline is ready.
    is_ready: bool,
}

impl RLCAggregationProver {
    /// Creates a new RLC aggregation prover.
    ///
    /// # Arguments
    /// * `max_batch_size` - Maximum number of proofs to aggregate (up to 32)
    /// * `k` - Circuit size parameter (2^k rows)
    pub fn new(max_batch_size: usize, k: u32) -> Self {
        let max_batch_size = max_batch_size.min(MAX_AGGREGATION_BATCH);

        // Create dummy circuit for keygen
        let dummy_circuit = SHPLONKAggregationCircuit::default();

        // Generate real keys
        let circuit_keys = generate_keys_for_circuit(
            &dummy_circuit,
            "shplonk_aggregation",
            1,
            k,
        );

        let pipeline = ProverPipeline::from_keys(
            crate::pipeline::PipelineConfig {
                k,
                self_verify: false,
                retry: crate::pipeline::RetryConfig::none(),
                proof_timeout: Some(Duration::from_secs(600)),
                verify_timeout: Some(Duration::from_secs(60)),
                deterministic_seed: None,
                enable_tracing: true,
            },
            circuit_keys.params,
            circuit_keys.pk,
            circuit_keys.vk,
        );

        Self {
            pipeline,
            max_batch_size,
            is_ready: true,
        }
    }

    /// Returns whether the prover is ready.
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Returns the maximum batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Aggregates multiple training proofs into a single proof.
    ///
    /// Each proof's public inputs must chain correctly:
    /// proof[i].new_hash == proof[i+1].old_hash.
    pub fn aggregate(
        &self,
        proofs: &[TrainingProofResultV2],
    ) -> Result<AggregatedTrainingProof, String> {
        if !self.is_ready {
            return Err("RLC aggregation prover not initialized".to_string());
        }
        if proofs.is_empty() {
            return Err("Cannot aggregate empty batch".to_string());
        }
        if proofs.len() > self.max_batch_size {
            return Err(format!(
                "Batch size {} exceeds maximum {}",
                proofs.len(),
                self.max_batch_size,
            ));
        }

        let start = Instant::now();

        // Build step witnesses from each proof's public inputs
        let steps: Vec<AggregationStepWitness> = proofs.iter()
            .map(|p| step_witness_from_public_inputs(&p.public_inputs))
            .collect();

        // Validate PI chaining
        for i in 0..steps.len() - 1 {
            if steps[i].new_hash_lo != steps[i + 1].old_hash_lo
                || steps[i].new_hash_hi != steps[i + 1].old_hash_hi
            {
                return Err(format!(
                    "PI chain broken between step {} and {}: \
                     new_hash != old_hash of next step",
                    i, i + 1,
                ));
            }
        }

        let witness = SHPLONKAggregationWitness::new(steps);
        let circuit = SHPLONKAggregationCircuit::new(witness.clone())
            .map_err(|e| format!("Failed to create aggregation circuit: {e}"))?;

        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Generate proof
        let proof_bytes = self.pipeline.prove(&circuit, &pi_refs)
            .map_err(|e| format!("RLC aggregation proof failed: {e}"))?;

        // Self-verify
        let verified = self.pipeline.verify(&proof_bytes, &pi_refs)
            .map_err(|e| format!("RLC aggregation self-verification failed: {e}"))?;

        if !verified {
            return Err("RLC aggregation proof failed self-verification".to_string());
        }

        let aggregation_time = start.elapsed();

        tracing::info!(
            num_steps = proofs.len(),
            proof_size = proof_bytes.len(),
            aggregation_time_ms = aggregation_time.as_millis() as u64,
            "RLC aggregation complete"
        );

        Ok(AggregatedTrainingProof {
            proof: proof_bytes,
            public_inputs: pi.clone(),
            num_steps: proofs.len(),
            total_loss: pi[4],
            total_error: pi[5],
            rlc_commitment: pi[7],
            first_old_hash: (pi[0], pi[1]),
            last_new_hash: (pi[2], pi[3]),
            individual_proofs: proofs.to_vec(),
            aggregation_time,
        })
    }

    /// Verifies an aggregated training proof.
    pub fn verify(&self, agg: &AggregatedTrainingProof) -> Result<bool, String> {
        if !self.is_ready {
            return Err("RLC aggregation prover not initialized".to_string());
        }

        let pi_refs: Vec<&[Fr]> = vec![&agg.public_inputs];
        self.pipeline.verify(&agg.proof, &pi_refs)
            .map_err(|e| format!("RLC aggregation verification failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_circuits::gadgets::poseidon::poseidon_hash_two;
    use helix_circuits::halo2_proofs::arithmetic::Field;

    /// Creates a mock TrainingProofResultV2 with given step number and chained hashes.
    fn make_chained_proof(
        step: u64,
        old_hash: (Fr, Fr),
        new_hash: (Fr, Fr),
    ) -> TrainingProofResultV2 {
        let loss = Fr::from(100u64);
        let error = Fr::from(5u64);
        let step_num = Fr::from(step);
        let checksum = Fr::from(42u64);

        let public_inputs = vec![
            old_hash.0,  // old_hash_lo
            old_hash.1,  // old_hash_hi
            new_hash.0,  // new_hash_lo
            new_hash.1,  // new_hash_hi
            loss,        // loss
            error,       // error_bound
            step_num,    // step_number
            checksum,    // error_checksum
        ];

        TrainingProofResultV2 {
            proof: vec![0xDE, 0xAD, 0xBE, 0xEF],
            public_inputs,
            loss,
            total_error: error,
            step_number: step,
            old_state_hash: old_hash,
            new_state_hash: new_hash,
            verified: true,
            generation_time: Duration::from_millis(10),
            verification_time: None,
            attempts: 1,
            from_cache: false,
            witness_hash: None,
        }
    }

    fn make_chained_proofs(n: usize) -> Vec<TrainingProofResultV2> {
        let mut proofs = Vec::with_capacity(n);
        for i in 0..n {
            let old_hash = (
                Fr::from((i * 10 + 1) as u64),
                Fr::from((i * 10 + 2) as u64),
            );
            let new_hash = (
                Fr::from(((i + 1) * 10 + 1) as u64),
                Fr::from(((i + 1) * 10 + 2) as u64),
            );
            proofs.push(make_chained_proof((i + 1) as u64, old_hash, new_hash));
        }
        proofs
    }

    #[test]
    fn test_rlc_aggregation_two_steps() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);
        assert!(prover.is_ready());

        let proofs = make_chained_proofs(2);
        let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

        assert_eq!(agg.num_steps, 2);
        assert_eq!(agg.public_inputs.len(), AGGREGATION_NUM_PUBLIC_INPUTS);
        assert!(!agg.proof.is_empty());

        // Verify
        let valid = prover.verify(&agg).expect("Verification should complete");
        assert!(valid, "Aggregated proof should verify");
    }

    #[test]
    fn test_rlc_aggregation_single_step() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);

        let proofs = make_chained_proofs(1);
        let agg = prover.aggregate(&proofs).expect("Single-step aggregation should succeed");

        assert_eq!(agg.num_steps, 1);
        let valid = prover.verify(&agg).expect("Verification should complete");
        assert!(valid);
    }

    #[test]
    fn test_rlc_chain_integrity() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);

        // Create proofs with broken chain
        let mut proofs = make_chained_proofs(2);
        // Break the chain by changing step[1]'s old_hash
        proofs[1].public_inputs[0] = Fr::from(9999u64);
        proofs[1].old_state_hash.0 = Fr::from(9999u64);

        let result = prover.aggregate(&proofs);
        assert!(result.is_err(), "Broken chain should be rejected");
    }

    #[test]
    fn test_rlc_error_accumulation() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);

        let proofs = make_chained_proofs(3);
        let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

        // total_error should be sum of individual errors (each is Fr::from(5))
        assert_eq!(agg.total_error, Fr::from(15u64));
        // total_loss should be sum of individual losses (each is Fr::from(100))
        assert_eq!(agg.total_loss, Fr::from(300u64));
    }

    #[test]
    fn test_rlc_rejects_empty() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);
        let result = prover.aggregate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rlc_boundary_hashes() {
        let _lock = crate::PROOF_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prover = RLCAggregationProver::new(16, 14);

        let proofs = make_chained_proofs(3);
        let agg = prover.aggregate(&proofs).expect("Aggregation should succeed");

        // first_old_hash should be step[0]'s old hash
        assert_eq!(agg.first_old_hash, (Fr::from(1u64), Fr::from(2u64)));
        // last_new_hash should be step[2]'s new hash
        assert_eq!(agg.last_new_hash, (Fr::from(31u64), Fr::from(32u64)));
    }
}
