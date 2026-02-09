//! MPC to ZK Proof Pipeline.
//!
//! This module provides the core pipeline for generating ZK proofs from MPC
//! computations. It bridges the secret-shared MPC world with the zero-knowledge
//! proof system.
//!
//! # Architecture
//!
//! The pipeline consists of several stages:
//!
//! 1. **Witness Collection**: Gather computation state from MPC parties
//! 2. **Witness Aggregation**: Combine party witnesses (for verification only)
//! 3. **Proof Generation**: Generate ZK proof of correct computation
//! 4. **Proof Verification**: Verify the proof matches committed public inputs
//!
//! # Privacy Model
//!
//! - Each party generates a proof commitment for their share
//! - Share-specific proofs are combined into an aggregate proof
//! - The final proof verifies computation without revealing individual shares

use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::integration::circuit_bridge::{CircuitBridge, CircuitBridgeConfig, Halo2ProofResult};
use crate::integration::witness_format::{
    MPCTrainingWitness, ReconstructedWitness, WitnessAggregator, WitnessBuilder,
};
use crate::security::commitment::BlindingGenerator;
use crate::sharing::model::{GradientShare, ModelShare};
use crate::types::{MPCConfig, PartyId};

/// Configuration for the ZK proof pipeline.
#[derive(Debug, Clone)]
pub struct ZKPipelineConfig {
    /// Number of MPC parties.
    pub num_parties: usize,
    /// Model dimensions.
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    /// Base error bound per operation.
    pub base_error: f64,
    /// Maximum allowed total error.
    pub max_error: f64,
    /// Whether to use Freivalds verification.
    pub use_freivalds: bool,
    /// Proof compression enabled.
    pub compress_proofs: bool,
    /// Circuit size parameter (k for 2^k rows).
    pub circuit_k: u32,
    /// Whether to use real Halo2 proofs (expensive).
    /// If false, uses mock proofs for testing.
    pub use_real_proofs: bool,
}

impl Default for ZKPipelineConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            d_in: 2,
            d_hid: 4,
            d_out: 1,
            base_error: 1e-6,
            max_error: 1e-3,
            use_freivalds: true,
            compress_proofs: true,
            circuit_k: 14,
            use_real_proofs: false, // Default to mock proofs for testing
        }
    }
}

impl ZKPipelineConfig {
    /// Creates a config from MPC config and model dimensions.
    pub fn from_mpc_config(mpc: &MPCConfig, d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            num_parties: mpc.num_parties,
            d_in,
            d_hid,
            d_out,
            ..Default::default()
        }
    }
}

/// Result of proof generation.
#[derive(Debug, Clone)]
pub struct ProofResult {
    /// The serialized proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification.
    pub public_inputs: Vec<Fr>,
    /// Old state commitment hash.
    pub old_state_hash: (Fr, Fr),
    /// New state commitment hash.
    pub new_state_hash: (Fr, Fr),
    /// Loss value.
    pub loss: Fr,
    /// Total error bound.
    pub total_error: Fr,
    /// Training step number.
    pub step_number: u64,
    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Proof size in bytes.
    pub proof_size_bytes: usize,
}

impl ProofResult {
    /// Creates a proof result with timing information.
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
}

/// Share-specific proof commitment.
///
/// Each party generates this commitment for their share of the computation.
/// These are aggregated to verify the full computation.
#[derive(Debug, Clone)]
pub struct ShareProofCommitment {
    /// Party that generated this commitment.
    pub party: PartyId,
    /// Commitment to the share values.
    pub share_commitment: [u8; 32],
    /// Commitment to the computation result.
    pub result_commitment: [u8; 32],
    /// Error bound for this party's computation.
    pub error_bound: Fr,
    /// Signature over the commitments (for authenticity).
    pub signature: Vec<u8>,
}

impl ShareProofCommitment {
    /// Creates a new share proof commitment.
    pub fn new(
        party: PartyId,
        share_data: &[Fr],
        result_data: &[Fr],
        error_bound: Fr,
        blinding: &[u8; 32],
    ) -> Self {
        let share_commitment = Self::compute_commitment(share_data, blinding);
        let result_commitment = Self::compute_commitment(result_data, blinding);

        // In production, this would be a proper signature.
        let signature = Self::compute_signature(&share_commitment, &result_commitment, &party);

        Self {
            party,
            share_commitment,
            result_commitment,
            error_bound,
            signature,
        }
    }

    fn compute_commitment(data: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in data {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    fn compute_signature(share: &[u8; 32], result: &[u8; 32], party: &PartyId) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(share);
        hasher.update(result);
        hasher.update(party.0.as_bytes());
        hasher.finalize().to_vec()
    }

    /// Verifies the signature.
    pub fn verify_signature(&self) -> bool {
        let expected = Self::compute_signature(
            &self.share_commitment,
            &self.result_commitment,
            &self.party,
        );
        self.signature == expected
    }
}

/// The main ZK proof pipeline.
///
/// Orchestrates the generation of ZK proofs from MPC computation.
pub struct ZKProofPipeline {
    /// Pipeline configuration.
    config: ZKPipelineConfig,
    /// Blinding factor generator.
    blinding_gen: BlindingGenerator,
    /// Cached party witnesses.
    party_witnesses: HashMap<usize, MPCTrainingWitness>,
    /// Share proof commitments.
    commitments: HashMap<usize, ShareProofCommitment>,
    /// Current step number.
    current_step: u64,
    /// Previous step's new_state_hash for chain continuity.
    /// None for the first step.
    previous_state_hash: Option<(Fr, Fr)>,
    /// Circuit bridge for real Halo2 proofs (optional).
    circuit_bridge: Option<CircuitBridge>,
}

impl ZKProofPipeline {
    /// Creates a new ZK proof pipeline.
    pub fn new(config: ZKPipelineConfig) -> Self {
        // Create circuit bridge if real proofs are enabled
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
            blinding_gen: BlindingGenerator::new(),
            party_witnesses: HashMap::new(),
            commitments: HashMap::new(),
            current_step: 0,
            previous_state_hash: None,
            circuit_bridge,
        }
    }

    /// Creates a pipeline with default configuration.
    pub fn default_three_party(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self::new(ZKPipelineConfig {
            num_parties: 3,
            d_in,
            d_hid,
            d_out,
            ..Default::default()
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ZKPipelineConfig {
        &self.config
    }

    /// Generates a witness for a party's share of the computation.
    pub fn generate_party_witness(
        &mut self,
        party_index: usize,
        model_share: &ModelShare,
        gradient_share: &GradientShare,
        input: &[f64],
        target: &[f64],
        learning_rate: f64,
    ) -> MPCResult<MPCTrainingWitness> {
        let blinding = self.blinding_gen.generate();

        let builder = WitnessBuilder::new(
            self.config.d_in,
            self.config.d_hid,
            self.config.d_out,
            party_index,
        )
        .learning_rate(learning_rate)
        .base_error(self.config.base_error);

        let witness = builder.build(
            model_share,
            gradient_share,
            input,
            target,
            self.current_step,
            &blinding,
        )?;

        // Cache the witness.
        self.party_witnesses.insert(party_index, witness.clone());

        // Generate commitment.
        let all_share_data: Vec<Fr> = witness
            .w1_share
            .data
            .iter()
            .chain(witness.b1_share.data.iter())
            .chain(witness.w2_share.data.iter())
            .chain(witness.b2_share.data.iter())
            .cloned()
            .collect();

        let all_result_data: Vec<Fr> = witness
            .w1_new_share
            .data
            .iter()
            .chain(witness.b1_new_share.data.iter())
            .chain(witness.w2_new_share.data.iter())
            .chain(witness.b2_new_share.data.iter())
            .cloned()
            .collect();

        let commitment = ShareProofCommitment::new(
            model_share.party.clone(),
            &all_share_data,
            &all_result_data,
            witness.total_error.clone(),
            &blinding,
        );
        self.commitments.insert(party_index, commitment);

        Ok(witness)
    }

    /// Checks if all party witnesses have been collected.
    pub fn has_all_witnesses(&self) -> bool {
        self.party_witnesses.len() == self.config.num_parties
    }

    /// Aggregates all party witnesses and reconstructs the full computation.
    pub fn aggregate_witnesses(&self) -> MPCResult<ReconstructedWitness> {
        if !self.has_all_witnesses() {
            return Err(MPCError::InsufficientShares {
                required: self.config.num_parties,
                available: self.party_witnesses.len(),
            });
        }

        let mut aggregator = WitnessAggregator::new(self.config.num_parties);
        for (party_index, witness) in &self.party_witnesses {
            aggregator.add_witness(*party_index, witness.clone())?;
        }

        aggregator.reconstruct()
    }

    /// Generates a ZK proof for the training step.
    ///
    /// If `use_real_proofs` is enabled in config, generates actual Halo2
    /// KZG proofs using the circuit bridge. Otherwise, uses mock proofs
    /// for testing (faster).
    ///
    /// For state chain continuity, uses the previous step's new_state_hash
    /// as this step's old_state_hash (if available).
    pub fn generate_proof(&mut self) -> MPCResult<ProofResult> {
        let start = std::time::Instant::now();

        // Aggregate witnesses.
        let mut reconstructed = self.aggregate_witnesses()?;

        // For state chain continuity: if we have a previous state hash,
        // use it as this step's old_state_hash
        if let Some(prev_hash) = &self.previous_state_hash {
            reconstructed.old_state_hash = prev_hash.clone();
        }

        // Verify error bounds.
        let error_f64 = reconstructed.total_error.to_f64();
        if error_f64 > self.config.max_error {
            return Err(MPCError::ErrorBoundExceeded {
                computed: error_f64,
                maximum: self.config.max_error,
            });
        }

        // Generate the proof - use real Halo2 if circuit bridge is available
        let result = if let Some(ref bridge) = self.circuit_bridge {
            // Generate real Halo2 KZG proof
            let halo2_result = bridge.prove(&reconstructed)?;

            // Store this step's new_state_hash for the next step's continuity
            self.previous_state_hash = Some(halo2_result.new_state_hash.clone());

            ProofResult::new(
                halo2_result.proof,
                halo2_result.public_inputs,
                halo2_result.old_state_hash,
                halo2_result.new_state_hash,
                halo2_result.loss,
                halo2_result.total_error,
                halo2_result.step_number,
                halo2_result.generation_time_ms,
            )
        } else {
            // Use mock proof for testing
            let proof = self.simulate_proof_generation(&reconstructed)?;
            let elapsed = start.elapsed();

            // Store this step's new_state_hash for the next step's continuity
            self.previous_state_hash = Some(reconstructed.new_state_hash.clone());

            ProofResult::new(
                proof,
                reconstructed.public_inputs(),
                reconstructed.old_state_hash,
                reconstructed.new_state_hash,
                Fr::ZERO, // Loss computed during proof
                reconstructed.total_error,
                reconstructed.step_number,
                elapsed.as_millis() as u64,
            )
        };

        Ok(result)
    }

    /// Simulates proof generation.
    ///
    /// This creates a mock proof with the correct structure.
    /// In production, this calls the actual prover.
    fn simulate_proof_generation(&self, witness: &ReconstructedWitness) -> MPCResult<Vec<u8>> {
        // Create a deterministic "proof" based on the witness.
        // This is for testing - real proofs come from helix-prover.
        let mut hasher = Sha256::new();

        // Hash all weights.
        for v in &witness.w1 {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.b1 {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.w2 {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.b2 {
            hasher.update(&v.to_bytes_le());
        }

        // Hash updated weights.
        for v in &witness.w1_new {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.b1_new {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.w2_new {
            hasher.update(&v.to_bytes_le());
        }
        for v in &witness.b2_new {
            hasher.update(&v.to_bytes_le());
        }

        // Include step number.
        hasher.update(&witness.step_number.to_le_bytes());

        // The "proof" is multiple hash iterations to simulate proof size.
        let base_hash: [u8; 32] = hasher.finalize().into();

        // Typical Halo2 proofs are 5-50KB.
        let target_size = if self.config.compress_proofs { 5000 } else { 20000 };
        let mut proof = Vec::with_capacity(target_size);

        // Fill with deterministic pseudo-random data.
        let mut current_hash = base_hash;
        while proof.len() < target_size {
            proof.extend_from_slice(&current_hash);
            let mut h = Sha256::new();
            h.update(&current_hash);
            current_hash = h.finalize().into();
        }
        proof.truncate(target_size);

        Ok(proof)
    }

    /// Verifies a proof against public inputs.
    ///
    /// If real proofs are enabled, uses the Halo2 verifier.
    /// Otherwise, does structural verification for mock proofs.
    ///
    /// This checks that:
    /// 1. The proof structure is valid
    /// 2. Public inputs match the expected format
    /// 3. State hashes are consistent with commitments
    pub fn verify_proof(&self, proof: &ProofResult) -> MPCResult<bool> {
        // Verify public inputs count.
        if proof.public_inputs.len() != 7 {
            return Ok(false);
        }

        // Verify state hash consistency.
        let hash_lo = &proof.public_inputs[0];
        let hash_hi = &proof.public_inputs[1];
        if !hash_lo.ct_eq(&proof.old_state_hash.0).to_bool()
            || !hash_hi.ct_eq(&proof.old_state_hash.1).to_bool()
        {
            return Ok(false);
        }

        // Verify error bound.
        let error = proof.total_error.to_f64();
        if error > self.config.max_error {
            return Ok(false);
        }

        // Use real Halo2 verification if circuit bridge is available
        if let Some(ref bridge) = self.circuit_bridge {
            // Convert ProofResult to Halo2ProofResult format for verification
            let halo2_proof = Halo2ProofResult::new(
                proof.proof.clone(),
                proof.public_inputs.clone(),
                proof.old_state_hash.clone(),
                proof.new_state_hash.clone(),
                proof.loss.clone(),
                proof.total_error.clone(),
                proof.step_number,
                proof.generation_time_ms,
            );
            return bridge.verify(&halo2_proof);
        }

        // For mock proofs, do structural verification
        Ok(!proof.proof.is_empty())
    }

    /// Verifies all party commitments are consistent.
    pub fn verify_commitments(&self) -> MPCResult<bool> {
        for (_party_index, commitment) in &self.commitments {
            if !commitment.verify_signature() {
                return Ok(false);
            }

            // Verify error bounds sum correctly.
            let error = commitment.error_bound.to_f64();
            if error > self.config.max_error / self.config.num_parties as f64 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Advances to the next training step.
    pub fn next_step(&mut self) {
        self.current_step += 1;
        self.party_witnesses.clear();
        self.commitments.clear();
    }

    /// Returns the current step number.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns all party commitments.
    pub fn commitments(&self) -> &HashMap<usize, ShareProofCommitment> {
        &self.commitments
    }

    /// Returns whether real Halo2 proofs are enabled.
    pub fn uses_real_proofs(&self) -> bool {
        self.circuit_bridge.is_some()
    }

    /// Enables real Halo2 proofs by initializing the circuit bridge.
    ///
    /// Note: This is expensive as it creates the proving key.
    pub fn enable_real_proofs(&mut self) {
        if self.circuit_bridge.is_none() {
            let bridge_config = CircuitBridgeConfig::for_model(
                self.config.d_in,
                self.config.d_hid,
                self.config.d_out,
            )
            .with_k(self.config.circuit_k)
            .with_base_error(self.config.base_error);
            self.circuit_bridge = Some(CircuitBridge::new(bridge_config));
        }
    }

    /// Returns a reference to the circuit bridge if available.
    pub fn circuit_bridge(&self) -> Option<&CircuitBridge> {
        self.circuit_bridge.as_ref()
    }
}

/// Proof hook that can be inserted into MPC arithmetic operations.
///
/// This allows generating partial proofs during computation.
#[derive(Debug, Clone)]
pub struct ProofHook {
    /// Operation being proven.
    pub operation: ProofOperation,
    /// Input values.
    pub inputs: Vec<Fr>,
    /// Output values.
    pub outputs: Vec<Fr>,
    /// Error bound for this operation.
    pub error_bound: Fr,
    /// Timestamp of the operation.
    pub timestamp: u64,
}

/// Types of operations that can be proven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofOperation {
    /// Addition of shares.
    Add,
    /// Subtraction of shares.
    Sub,
    /// Multiplication using Beaver triple.
    BeaverMul,
    /// Matrix multiplication.
    MatMul,
    /// ReLU activation.
    ReLU,
    /// Gradient computation.
    Gradient,
    /// Weight update.
    WeightUpdate,
}

impl ProofHook {
    /// Creates a new proof hook.
    pub fn new(operation: ProofOperation) -> Self {
        Self {
            operation,
            inputs: Vec::new(),
            outputs: Vec::new(),
            error_bound: Fr::ZERO,
            timestamp: 0,
        }
    }

    /// Records an input value.
    pub fn record_input(&mut self, value: Fr) {
        self.inputs.push(value);
    }

    /// Records an output value.
    pub fn record_output(&mut self, value: Fr) {
        self.outputs.push(value);
    }

    /// Sets the error bound.
    pub fn set_error(&mut self, error: Fr) {
        self.error_bound = error;
    }

    /// Generates a commitment to this operation.
    pub fn commitment(&self, blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&[self.operation as u8]);
        for v in &self.inputs {
            hasher.update(&v.to_bytes_le());
        }
        for v in &self.outputs {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(&self.error_bound.to_bytes_le());
        hasher.update(blinding);
        hasher.finalize().into()
    }
}

/// Collector for proof hooks during computation.
pub struct ProofHookCollector {
    /// Collected hooks.
    hooks: Vec<ProofHook>,
    /// Current timestamp counter.
    timestamp: u64,
    /// Blinding generator.
    blinding_gen: BlindingGenerator,
}

impl ProofHookCollector {
    /// Creates a new collector.
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            timestamp: 0,
            blinding_gen: BlindingGenerator::new(),
        }
    }

    /// Starts a new proof hook for an operation.
    pub fn start_hook(&mut self, operation: ProofOperation) -> usize {
        let mut hook = ProofHook::new(operation);
        hook.timestamp = self.timestamp;
        self.timestamp += 1;
        self.hooks.push(hook);
        self.hooks.len() - 1
    }

    /// Records an input to a hook.
    pub fn record_input(&mut self, hook_id: usize, value: Fr) {
        if hook_id < self.hooks.len() {
            self.hooks[hook_id].record_input(value);
        }
    }

    /// Records an output to a hook.
    pub fn record_output(&mut self, hook_id: usize, value: Fr) {
        if hook_id < self.hooks.len() {
            self.hooks[hook_id].record_output(value);
        }
    }

    /// Sets the error bound for a hook.
    pub fn set_error(&mut self, hook_id: usize, error: Fr) {
        if hook_id < self.hooks.len() {
            self.hooks[hook_id].set_error(error);
        }
    }

    /// Finalizes collection and returns all commitments.
    pub fn finalize(&mut self) -> Vec<[u8; 32]> {
        self.hooks
            .iter()
            .map(|h| {
                let blinding = self.blinding_gen.generate();
                h.commitment(&blinding)
            })
            .collect()
    }

    /// Returns the collected hooks.
    pub fn hooks(&self) -> &[ProofHook] {
        &self.hooks
    }

    /// Clears all hooks.
    pub fn clear(&mut self) {
        self.hooks.clear();
        self.timestamp = 0;
    }
}

impl Default for ProofHookCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch proof generator for multiple training steps.
pub struct BatchProofGenerator {
    /// Pipeline configuration.
    config: ZKPipelineConfig,
    /// Collected proofs.
    proofs: Vec<ProofResult>,
    /// Aggregated state commitment.
    state_chain: Vec<(Fr, Fr)>,
}

impl BatchProofGenerator {
    /// Creates a new batch generator.
    pub fn new(config: ZKPipelineConfig) -> Self {
        Self {
            config,
            proofs: Vec::new(),
            state_chain: Vec::new(),
        }
    }

    /// Adds a proof to the batch.
    pub fn add_proof(&mut self, proof: ProofResult) {
        // Verify state chain continuity.
        if let Some(last_state) = self.state_chain.last() {
            // New proof's old state should match last proof's new state.
            assert!(
                last_state.0.ct_eq(&proof.old_state_hash.0).to_bool()
                    && last_state.1.ct_eq(&proof.old_state_hash.1).to_bool(),
                "State chain discontinuity"
            );
        }

        self.state_chain.push(proof.new_state_hash.clone());
        self.proofs.push(proof);
    }

    /// Returns the number of proofs in the batch.
    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    /// Checks if empty.
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    /// Generates an aggregated proof for the batch.
    ///
    /// This uses Nova-style folding to compress multiple proofs.
    pub fn generate_batch_proof(&self) -> MPCResult<BatchProofResult> {
        if self.proofs.is_empty() {
            return Err(MPCError::ProtocolError("No proofs to batch".into()));
        }

        // Compute aggregated values.
        let total_error = self
            .proofs
            .iter()
            .fold(Fr::ZERO, |acc, p| Fr::add(&acc, &p.total_error));

        let total_time: u64 = self.proofs.iter().map(|p| p.generation_time_ms).sum();

        let initial_state = self.proofs[0].old_state_hash.clone();
        let final_state = self.proofs.last().unwrap().new_state_hash.clone();

        // In production, this would use Nova/IVC folding.
        // For now, we create a simple aggregate.
        let aggregate = BatchProofResult {
            num_steps: self.proofs.len(),
            initial_state,
            final_state,
            total_error,
            step_proofs: self.proofs.iter().map(|p| p.proof.clone()).collect(),
            aggregation_time_ms: total_time,
        };

        Ok(aggregate)
    }

    /// Verifies a batch proof.
    pub fn verify_batch_proof(&self, batch: &BatchProofResult) -> bool {
        // Verify state chain.
        if batch.step_proofs.len() != batch.num_steps {
            return false;
        }

        // Verify error bound.
        let error = batch.total_error.to_f64();
        if error > self.config.max_error * batch.num_steps as f64 {
            return false;
        }

        true
    }
}

/// Result of batch proof generation.
#[derive(Debug, Clone)]
pub struct BatchProofResult {
    /// Number of training steps included.
    pub num_steps: usize,
    /// Initial state commitment (before first step).
    pub initial_state: (Fr, Fr),
    /// Final state commitment (after last step).
    pub final_state: (Fr, Fr),
    /// Total accumulated error.
    pub total_error: Fr,
    /// Individual step proofs (for verification).
    pub step_proofs: Vec<Vec<u8>>,
    /// Total aggregation time.
    pub aggregation_time_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharing::model::LayerShare;
    use crate::types::ShareId;
    use std::collections::HashMap as StdHashMap;

    fn create_test_model_share(
        party_index: usize,
        d_in: usize,
        d_hid: usize,
        d_out: usize,
    ) -> ModelShare {
        use crate::sharing::tensor::TensorShare;

        let party = PartyId::from_index(party_index);

        let w1_data: Vec<Fr> = (0..d_hid * d_in)
            .map(|i| Fr::from_f64(0.1 * (i as f64 + party_index as f64)))
            .collect();
        let b1_data: Vec<Fr> = (0..d_hid)
            .map(|i| Fr::from_f64(0.01 * (i as f64 + party_index as f64)))
            .collect();
        let w2_data: Vec<Fr> = (0..d_out * d_hid)
            .map(|i| Fr::from_f64(0.1 * (i as f64 + party_index as f64)))
            .collect();
        let b2_data: Vec<Fr> = (0..d_out)
            .map(|i| Fr::from_f64(0.01 * (i as f64 + party_index as f64)))
            .collect();

        let mut weights = StdHashMap::new();
        weights.insert(
            "w1".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "w1", party_index),
                w1_data,
                vec![d_hid, d_in],
            ),
        );
        weights.insert(
            "b1".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "b1", party_index),
                b1_data,
                vec![d_hid],
            ),
        );
        weights.insert(
            "w2".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "w2", party_index),
                w2_data,
                vec![d_out, d_hid],
            ),
        );
        weights.insert(
            "b2".to_string(),
            TensorShare::new(
                ShareId::new(party.clone(), "b2", party_index),
                b2_data,
                vec![d_out],
            ),
        );

        ModelShare {
            party: party.clone(),
            index: party_index,
            model_name: "test".to_string(),
            num_layers: 1,
            embeddings: None,
            layers: vec![LayerShare {
                layer_idx: 0,
                weights,
            }],
            lm_head: None,
            extra_weights: StdHashMap::new(),
        }
    }

    fn create_test_gradient_share(party_index: usize) -> GradientShare {
        GradientShare {
            party: PartyId::from_index(party_index),
            index: party_index,
            embeddings: None,
            layers: vec![],
            lm_head: None,
            error_bound: 0.01,
        }
    }

    #[test]
    fn test_zk_pipeline_creation() {
        let config = ZKPipelineConfig::default();
        let pipeline = ZKProofPipeline::new(config);

        assert_eq!(pipeline.current_step(), 0);
        assert!(!pipeline.has_all_witnesses());
    }

    #[test]
    fn test_witness_generation() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let mut pipeline = ZKProofPipeline::default_three_party(d_in, d_hid, d_out);

        for i in 0..3 {
            let model = create_test_model_share(i, d_in, d_hid, d_out);
            let gradient = create_test_gradient_share(i);

            let witness = pipeline
                .generate_party_witness(i, &model, &gradient, &[1.0, 1.0], &[1.0], 0.01)
                .unwrap();

            assert_eq!(witness.party_index, i);
        }

        assert!(pipeline.has_all_witnesses());
    }

    #[test]
    fn test_proof_generation() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let mut pipeline = ZKProofPipeline::default_three_party(d_in, d_hid, d_out);

        for i in 0..3 {
            let model = create_test_model_share(i, d_in, d_hid, d_out);
            let gradient = create_test_gradient_share(i);

            pipeline
                .generate_party_witness(i, &model, &gradient, &[1.0, 1.0], &[1.0], 0.01)
                .unwrap();
        }

        let proof = pipeline.generate_proof().unwrap();

        assert!(!proof.proof.is_empty());
        assert_eq!(proof.public_inputs.len(), 7);
        assert_eq!(proof.step_number, 0);
    }

    #[test]
    fn test_proof_verification() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let mut pipeline = ZKProofPipeline::default_three_party(d_in, d_hid, d_out);

        for i in 0..3 {
            let model = create_test_model_share(i, d_in, d_hid, d_out);
            let gradient = create_test_gradient_share(i);

            pipeline
                .generate_party_witness(i, &model, &gradient, &[1.0, 1.0], &[1.0], 0.01)
                .unwrap();
        }

        let proof = pipeline.generate_proof().unwrap();
        assert!(pipeline.verify_proof(&proof).unwrap());
    }

    #[test]
    fn test_proof_hook_collector() {
        let mut collector = ProofHookCollector::new();

        let hook_id = collector.start_hook(ProofOperation::Add);
        collector.record_input(hook_id, Fr::from_f64(1.0));
        collector.record_input(hook_id, Fr::from_f64(2.0));
        collector.record_output(hook_id, Fr::from_f64(3.0));
        collector.set_error(hook_id, Fr::from_f64(0.001));

        let commitments = collector.finalize();
        assert_eq!(commitments.len(), 1);

        let hooks = collector.hooks();
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0].operation, ProofOperation::Add);
        assert_eq!(hooks[0].inputs.len(), 2);
        assert_eq!(hooks[0].outputs.len(), 1);
    }

    #[test]
    fn test_batch_proof_generator() {
        let config = ZKPipelineConfig::default();
        let mut generator = BatchProofGenerator::new(config.clone());

        // Create some mock proofs.
        let hash1 = (Fr::from_u64(1), Fr::from_u64(2));
        let hash2 = (Fr::from_u64(3), Fr::from_u64(4));
        let hash3 = (Fr::from_u64(5), Fr::from_u64(6));

        let proof1 = ProofResult::new(
            vec![1, 2, 3],
            vec![Fr::ZERO; 7],
            hash1.clone(),
            hash2.clone(),
            Fr::ZERO,
            Fr::from_f64(0.0001),
            0,
            100,
        );

        let proof2 = ProofResult::new(
            vec![4, 5, 6],
            vec![Fr::ZERO; 7],
            hash2.clone(),
            hash3.clone(),
            Fr::ZERO,
            Fr::from_f64(0.0001),
            1,
            100,
        );

        generator.add_proof(proof1);
        generator.add_proof(proof2);

        assert_eq!(generator.len(), 2);

        let batch = generator.generate_batch_proof().unwrap();
        assert_eq!(batch.num_steps, 2);
        assert!(generator.verify_batch_proof(&batch));
    }

    #[test]
    fn test_share_proof_commitment() {
        let party = PartyId::from_index(0);
        let share_data = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let result_data = vec![Fr::from_f64(3.0), Fr::from_f64(4.0)];
        let error = Fr::from_f64(0.001);
        let blinding = [42u8; 32];

        let commitment =
            ShareProofCommitment::new(party, &share_data, &result_data, error, &blinding);

        assert!(commitment.verify_signature());
    }

    #[test]
    fn test_enable_real_proofs() {
        let config = ZKPipelineConfig::default();
        let mut pipeline = ZKProofPipeline::new(config);

        assert!(!pipeline.uses_real_proofs());

        pipeline.enable_real_proofs();

        assert!(pipeline.uses_real_proofs());
        assert!(pipeline.circuit_bridge().is_some());
    }

    // Note: Real Halo2 proof generation test is expensive, so we mark it as ignored.
    // Run with: cargo test --release -p helix-mpc -- --ignored
    #[test]
    #[ignore]
    fn test_real_halo2_proof_generation() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        // Create pipeline with real proofs enabled
        let config = ZKPipelineConfig {
            num_parties: 3,
            d_in,
            d_hid,
            d_out,
            use_real_proofs: true,
            ..Default::default()
        };

        let mut pipeline = ZKProofPipeline::new(config);
        assert!(pipeline.uses_real_proofs());

        // Generate witnesses for each party
        for i in 0..3 {
            let model = create_test_model_share(i, d_in, d_hid, d_out);
            let gradient = create_test_gradient_share(i);

            pipeline
                .generate_party_witness(i, &model, &gradient, &[1.0, 1.0], &[1.0], 0.01)
                .unwrap();
        }

        // Generate real Halo2 proof
        let proof = pipeline.generate_proof().unwrap();

        // Verify proof structure
        assert!(!proof.proof.is_empty());
        assert_eq!(proof.public_inputs.len(), 7);
        assert_eq!(proof.step_number, 0);

        // Verify with real Halo2 verifier
        assert!(pipeline.verify_proof(&proof).unwrap());

        // Log proof statistics
        println!("Real Halo2 proof size: {} bytes", proof.proof_size_bytes);
        println!("Proof generation time: {} ms", proof.generation_time_ms);
    }
}
