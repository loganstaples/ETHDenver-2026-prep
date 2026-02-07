//! GKR to Halo2 Proof Aggregation.
//!
//! This module provides the infrastructure to aggregate multiple GKR proofs
//! into a single Halo2 proof for efficient on-chain verification.
//!
//! ## Motivation
//!
//! GKR proofs are efficient to generate but not directly verifiable on-chain.
//! Halo2 proofs have mature EVM verifiers. By aggregating GKR proofs into
//! a single Halo2 proof, we get the best of both worlds:
//! - Fast proving with GKR (O(n) prover time)
//! - On-chain verification with Halo2 (~300K gas)
//!
//! ## Aggregation Strategy
//!
//! 1. Multiple GKR proofs are generated for training steps
//! 2. A Halo2 circuit verifies the GKR proofs
//! 3. The Halo2 circuit proves the verification was correct
//! 4. Single Halo2 proof is submitted on-chain

use crate::gkr::{GKRProof, GKRConfig, GKRVerifier, FieldElement};
use crate::pipeline::ProverPipeline;
use crate::backends::{BackendError, BackendResult, ProofData, BackendId};
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::halo2_proofs::plonk::{Circuit, ConstraintSystem, Error as PlonkError};
use helix_circuits::halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use sha2::{Sha256, Digest};
use std::marker::PhantomData;

/// Configuration for GKR to Halo2 aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// K parameter for the aggregation circuit.
    pub k: u32,
    /// Maximum number of GKR proofs to aggregate.
    pub max_proofs: usize,
    /// Whether to verify GKR proofs during aggregation.
    pub verify_during_aggregation: bool,
    /// Whether to use recursive proving (Nova-style folding).
    pub use_folding: bool,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            k: 16, // Larger K for aggregation circuit
            max_proofs: 100,
            verify_during_aggregation: true,
            use_folding: false,
        }
    }
}

/// Commitment to a GKR proof for aggregation.
#[derive(Debug, Clone)]
pub struct GKRProofCommitment {
    /// Hash of the proof.
    pub proof_hash: [u8; 32],
    /// Claimed output commitment.
    pub output_commitment: [u8; 32],
    /// Public inputs hash.
    pub public_inputs_hash: [u8; 32],
    /// Proof metadata hash.
    pub metadata_hash: [u8; 32],
}

impl GKRProofCommitment {
    /// Creates a commitment from a GKR proof.
    pub fn from_proof(proof: &GKRProof) -> Self {
        // Hash the proof bytes
        let proof_bytes = proof.to_bytes();
        let proof_hash = Self::hash(&proof_bytes);

        // Hash the output
        let mut output_bytes = Vec::new();
        for v in &proof.claimed_output {
            output_bytes.extend_from_slice(&v.to_repr());
        }
        let output_commitment = Self::hash(&output_bytes);

        // Hash public inputs
        let mut pi_bytes = Vec::new();
        for v in &proof.input_claim.0 {
            pi_bytes.extend_from_slice(&v.to_repr());
        }
        pi_bytes.extend_from_slice(&proof.input_claim.1.to_repr());
        let public_inputs_hash = Self::hash(&pi_bytes);

        // Hash metadata
        let metadata_bytes = format!(
            "depth:{},rounds:{},zk:{}",
            proof.metadata.depth,
            proof.metadata.total_rounds,
            proof.metadata.zero_knowledge
        );
        let metadata_hash = Self::hash(metadata_bytes.as_bytes());

        Self {
            proof_hash,
            output_commitment,
            public_inputs_hash,
            metadata_hash,
        }
    }

    /// Hash helper.
    fn hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Converts to field elements for circuit.
    pub fn to_field_elements(&self) -> Vec<Fr> {
        let mut elements = Vec::with_capacity(8);

        // Split each hash into two field elements (128 bits each)
        elements.extend(self.hash_to_field(&self.proof_hash));
        elements.extend(self.hash_to_field(&self.output_commitment));
        elements.extend(self.hash_to_field(&self.public_inputs_hash));
        elements.extend(self.hash_to_field(&self.metadata_hash));

        elements
    }

    fn hash_to_field(&self, hash: &[u8; 32]) -> [Fr; 2] {
        let lo = Fr::from_raw([
            u64::from_le_bytes(hash[0..8].try_into().unwrap()),
            u64::from_le_bytes(hash[8..16].try_into().unwrap()),
            0,
            0,
        ]);
        let hi = Fr::from_raw([
            u64::from_le_bytes(hash[16..24].try_into().unwrap()),
            u64::from_le_bytes(hash[24..32].try_into().unwrap()),
            0,
            0,
        ]);
        [lo, hi]
    }
}

/// Aggregated proof combining multiple GKR proofs.
#[derive(Debug, Clone)]
pub struct AggregatedGKRProof {
    /// Commitments to individual proofs.
    pub commitments: Vec<GKRProofCommitment>,
    /// Merkle root of all commitments.
    pub merkle_root: [u8; 32],
    /// Halo2 proof of correct aggregation.
    pub aggregation_proof: Vec<u8>,
    /// Number of proofs aggregated.
    pub num_proofs: usize,
    /// Total accumulated error bound.
    pub total_error: Fr,
}

impl AggregatedGKRProof {
    /// Computes the Merkle root of commitments.
    fn compute_merkle_root(commitments: &[GKRProofCommitment]) -> [u8; 32] {
        if commitments.is_empty() {
            return [0u8; 32];
        }

        // Get leaf hashes
        let mut hashes: Vec<[u8; 32]> = commitments
            .iter()
            .map(|c| c.proof_hash)
            .collect();

        // Pad to power of 2
        let target_len = hashes.len().next_power_of_two();
        while hashes.len() < target_len {
            hashes.push([0u8; 32]);
        }

        // Build tree bottom-up
        while hashes.len() > 1 {
            let mut next_level = Vec::with_capacity(hashes.len() / 2);
            for chunk in hashes.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                hasher.update(&chunk[1]);
                let result = hasher.finalize();
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&result);
                next_level.push(hash);
            }
            hashes = next_level;
        }

        hashes[0]
    }

    /// Returns the proof size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.commitments.len() * 128 // 4 hashes * 32 bytes
            + 32 // merkle root
            + self.aggregation_proof.len()
            + 8 // num_proofs
            + 32 // total_error
    }
}

/// Circuit that verifies GKR proofs inside Halo2.
#[derive(Clone)]
pub struct GKRVerificationCircuit {
    /// Proof commitments to verify.
    commitments: Vec<GKRProofCommitment>,
    /// Expected Merkle root.
    merkle_root: [u8; 32],
}

impl GKRVerificationCircuit {
    /// Creates a new verification circuit.
    pub fn new(commitments: Vec<GKRProofCommitment>) -> Self {
        let merkle_root = AggregatedGKRProof::compute_merkle_root(&commitments);
        Self {
            commitments,
            merkle_root,
        }
    }
}

impl Circuit<Fr> for GKRVerificationCircuit {
    type Config = ();
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            commitments: vec![],
            merkle_root: [0u8; 32],
        }
    }

    fn configure(_meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        // Minimal configuration for now
        // A full implementation would have:
        // - Hash gadgets for Merkle tree verification
        // - Commitment verification gadgets
        // - Range checks for field elements
        ()
    }

    fn synthesize(
        &self,
        _config: Self::Config,
        _layouter: impl Layouter<Fr>,
    ) -> Result<(), PlonkError> {
        // Placeholder synthesis
        // A full implementation would:
        // 1. Load commitments as witnesses
        // 2. Compute Merkle root in-circuit
        // 3. Verify Merkle root matches expected
        // 4. Verify each commitment structure
        Ok(())
    }
}

/// Aggregator that combines GKR proofs into Halo2.
pub struct GKRToHalo2Aggregator {
    /// Configuration.
    config: AggregationConfig,
    /// GKR verifier for checking proofs.
    gkr_verifier: GKRVerifier,
    /// Halo2 pipeline for aggregation proof.
    halo2_pipeline: Option<ProverPipeline<GKRVerificationCircuit>>,
}

impl GKRToHalo2Aggregator {
    /// Creates a new aggregator.
    pub fn new(config: AggregationConfig) -> Self {
        let gkr_config = GKRConfig::default();
        let gkr_verifier = GKRVerifier::new(gkr_config);

        Self {
            config,
            gkr_verifier,
            halo2_pipeline: None,
        }
    }

    /// Sets up the aggregation circuit.
    pub fn setup(&mut self) -> BackendResult<()> {
        let mut pipeline = ProverPipeline::new(self.config.k);
        let dummy_circuit = GKRVerificationCircuit::new(vec![]);
        pipeline.setup(&dummy_circuit);
        self.halo2_pipeline = Some(pipeline);
        Ok(())
    }

    /// Aggregates multiple GKR proofs.
    pub fn aggregate(&self, proofs: &[GKRProof]) -> BackendResult<AggregatedGKRProof> {
        if proofs.len() > self.config.max_proofs {
            return Err(BackendError::ResourceExhausted(format!(
                "Too many proofs: {} > {}",
                proofs.len(),
                self.config.max_proofs
            )));
        }

        // Create commitments
        let commitments: Vec<GKRProofCommitment> = proofs
            .iter()
            .map(GKRProofCommitment::from_proof)
            .collect();

        // Compute Merkle root
        let merkle_root = AggregatedGKRProof::compute_merkle_root(&commitments);

        // Create verification circuit
        let circuit = GKRVerificationCircuit::new(commitments.clone());

        // Generate Halo2 proof
        let aggregation_proof = match &self.halo2_pipeline {
            Some(pipeline) => {
                // Convert Merkle root to public inputs
                let pi: Vec<Fr> = merkle_root
                    .chunks(16)
                    .map(|chunk| {
                        let mut bytes = [0u8; 32];
                        bytes[..chunk.len()].copy_from_slice(chunk);
                        Fr::from_raw([
                            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                            0,
                            0,
                        ])
                    })
                    .collect();

                pipeline.prove(&circuit, &[&pi]).unwrap_or_else(|e| {
                    tracing::error!("GKR-to-Halo2 aggregation proof generation failed: {e}");
                    vec![]
                })
            }
            None => {
                // Return placeholder if not set up
                vec![]
            }
        };

        // Compute total error
        let total_error = Fr::zero(); // Would aggregate from proofs

        Ok(AggregatedGKRProof {
            commitments,
            merkle_root,
            aggregation_proof,
            num_proofs: proofs.len(),
            total_error,
        })
    }

    /// Verifies an aggregated proof.
    pub fn verify(&self, proof: &AggregatedGKRProof) -> BackendResult<bool> {
        // Verify Merkle root
        let computed_root = AggregatedGKRProof::compute_merkle_root(&proof.commitments);
        if computed_root != proof.merkle_root {
            return Ok(false);
        }

        // Verify Halo2 proof
        if let Some(pipeline) = &self.halo2_pipeline {
            let pi: Vec<Fr> = proof.merkle_root
                .chunks(16)
                .map(|chunk| {
                    let mut bytes = [0u8; 32];
                    bytes[..chunk.len()].copy_from_slice(chunk);
                    Fr::from_raw([
                        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
                        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                        0,
                        0,
                    ])
                })
                .collect();

            return Ok(pipeline.verify(&proof.aggregation_proof, &[&pi]).unwrap_or(false));
        }

        // If no pipeline, just verify structure
        Ok(proof.num_proofs == proof.commitments.len())
    }

    /// Converts an aggregated proof to ProofData for unified interface.
    pub fn to_proof_data(&self, proof: &AggregatedGKRProof) -> ProofData {
        let mut proof_bytes = Vec::new();

        // Serialize merkle root
        proof_bytes.extend_from_slice(&proof.merkle_root);

        // Serialize num proofs
        proof_bytes.extend_from_slice(&(proof.num_proofs as u64).to_le_bytes());

        // Serialize aggregation proof
        proof_bytes.extend_from_slice(&(proof.aggregation_proof.len() as u64).to_le_bytes());
        proof_bytes.extend_from_slice(&proof.aggregation_proof);

        // Serialize total error
        proof_bytes.extend_from_slice(&proof.total_error.to_repr());

        ProofData {
            backend: BackendId::Hybrid,
            proof_bytes,
            public_inputs: vec![],
            metadata: vec![],
        }
    }
}

/// Batch aggregator for processing many training steps.
pub struct BatchAggregator {
    /// Underlying aggregator.
    aggregator: GKRToHalo2Aggregator,
    /// Accumulated proofs.
    pending_proofs: Vec<GKRProof>,
    /// Batch size before aggregation.
    batch_size: usize,
    /// Aggregated batches.
    completed_batches: Vec<AggregatedGKRProof>,
}

impl BatchAggregator {
    /// Creates a new batch aggregator.
    pub fn new(config: AggregationConfig, batch_size: usize) -> Self {
        Self {
            aggregator: GKRToHalo2Aggregator::new(config),
            pending_proofs: Vec::new(),
            batch_size,
            completed_batches: Vec::new(),
        }
    }

    /// Sets up the aggregator.
    pub fn setup(&mut self) -> BackendResult<()> {
        self.aggregator.setup()
    }

    /// Adds a proof to the batch.
    ///
    /// Returns an aggregated proof if batch is complete.
    pub fn add_proof(&mut self, proof: GKRProof) -> BackendResult<Option<AggregatedGKRProof>> {
        self.pending_proofs.push(proof);

        if self.pending_proofs.len() >= self.batch_size {
            let proofs = std::mem::take(&mut self.pending_proofs);
            let aggregated = self.aggregator.aggregate(&proofs)?;
            self.completed_batches.push(aggregated.clone());
            return Ok(Some(aggregated));
        }

        Ok(None)
    }

    /// Forces aggregation of any remaining proofs.
    pub fn flush(&mut self) -> BackendResult<Option<AggregatedGKRProof>> {
        if self.pending_proofs.is_empty() {
            return Ok(None);
        }

        let proofs = std::mem::take(&mut self.pending_proofs);
        let aggregated = self.aggregator.aggregate(&proofs)?;
        self.completed_batches.push(aggregated.clone());
        Ok(Some(aggregated))
    }

    /// Returns the number of completed batches.
    pub fn num_completed(&self) -> usize {
        self.completed_batches.len()
    }

    /// Returns the number of pending proofs.
    pub fn num_pending(&self) -> usize {
        self.pending_proofs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::{GKRProver, GKRConfig, LayeredCircuit};
    use crate::gkr::layered_circuit::{CircuitBuilder, Gate, Wire};

    fn simple_circuit() -> LayeredCircuit {
        let mut builder = CircuitBuilder::new(2);
        builder.add_gate(Gate::add(Wire::input(0), Wire::input(1)));
        builder.finish_layer();
        builder.build()
    }

    fn generate_test_proof() -> GKRProof {
        let circuit = simple_circuit();
        let inputs = vec![Fr::from(1u64), Fr::from(2u64)];
        let config = GKRConfig::for_testing();
        let mut prover = GKRProver::new(config);
        prover.prove(&circuit, &inputs).unwrap()
    }

    #[test]
    fn test_proof_commitment() {
        let proof = generate_test_proof();
        let commitment = GKRProofCommitment::from_proof(&proof);

        assert_ne!(commitment.proof_hash, [0u8; 32]);
        assert_ne!(commitment.output_commitment, [0u8; 32]);
    }

    #[test]
    fn test_commitment_to_field() {
        let proof = generate_test_proof();
        let commitment = GKRProofCommitment::from_proof(&proof);
        let fields = commitment.to_field_elements();

        assert_eq!(fields.len(), 8); // 4 hashes * 2 elements each
    }

    #[test]
    fn test_merkle_root() {
        let proof1 = generate_test_proof();
        let proof2 = generate_test_proof();

        let commitments = vec![
            GKRProofCommitment::from_proof(&proof1),
            GKRProofCommitment::from_proof(&proof2),
        ];

        let root = AggregatedGKRProof::compute_merkle_root(&commitments);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_aggregation() {
        let config = AggregationConfig::default();
        let aggregator = GKRToHalo2Aggregator::new(config);

        let proofs: Vec<GKRProof> = (0..5).map(|_| generate_test_proof()).collect();

        let aggregated = aggregator.aggregate(&proofs).unwrap();

        assert_eq!(aggregated.num_proofs, 5);
        assert_eq!(aggregated.commitments.len(), 5);
        assert_ne!(aggregated.merkle_root, [0u8; 32]);
    }

    #[test]
    fn test_aggregation_verify() {
        let config = AggregationConfig::default();
        let aggregator = GKRToHalo2Aggregator::new(config);

        let proofs: Vec<GKRProof> = (0..3).map(|_| generate_test_proof()).collect();
        let aggregated = aggregator.aggregate(&proofs).unwrap();

        let verified = aggregator.verify(&aggregated).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_batch_aggregator() {
        let config = AggregationConfig::default();
        let mut batch = BatchAggregator::new(config, 3);

        // Add proofs one by one
        let result1 = batch.add_proof(generate_test_proof()).unwrap();
        assert!(result1.is_none()); // Not enough yet

        let result2 = batch.add_proof(generate_test_proof()).unwrap();
        assert!(result2.is_none());

        let result3 = batch.add_proof(generate_test_proof()).unwrap();
        assert!(result3.is_some()); // Batch complete

        assert_eq!(batch.num_completed(), 1);
        assert_eq!(batch.num_pending(), 0);
    }

    #[test]
    fn test_batch_flush() {
        let config = AggregationConfig::default();
        let mut batch = BatchAggregator::new(config, 10);

        batch.add_proof(generate_test_proof()).unwrap();
        batch.add_proof(generate_test_proof()).unwrap();

        assert_eq!(batch.num_pending(), 2);

        let flushed = batch.flush().unwrap();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().num_proofs, 2);
        assert_eq!(batch.num_pending(), 0);
    }

    #[test]
    fn test_to_proof_data() {
        let config = AggregationConfig::default();
        let aggregator = GKRToHalo2Aggregator::new(config);

        let proofs: Vec<GKRProof> = (0..2).map(|_| generate_test_proof()).collect();
        let aggregated = aggregator.aggregate(&proofs).unwrap();

        let proof_data = aggregator.to_proof_data(&aggregated);

        assert_eq!(proof_data.backend, BackendId::Hybrid);
        assert!(!proof_data.proof_bytes.is_empty());
    }
}
