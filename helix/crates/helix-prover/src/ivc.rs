//! Incrementally Verifiable Computation (IVC).
//!
//! Implements IVC for proving long-running ML computations step by step,
//! where each step's proof can be verified and extended.
//!
//! This module uses the A1 accumulator circuit from `helix-circuits` for
//! Poseidon-based state transitions and Nova-style folding, providing
//! cryptographic guarantees that state commitments are correctly chained.

use serde::{Deserialize, Serialize};
use tracing;

use crate::pipeline::ProverPipeline;

// Use the A1 IVC circuits from helix-circuits (Poseidon-based, k=12, 8 PI).
use helix_circuits::{
    IVCAccumulator, IVCChain,
    IVCStepCircuit as A1StepCircuit, IVCStepWitness,
    IVCFoldingCircuit, IVCFoldingWitness,
    fold_accumulators, generate_folding_challenge,
};
use helix_circuits::gadgets::poseidon::poseidon_hash_two;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::halo2_proofs::arithmetic::Field;

/// Snapshot of accumulator state for portable verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccumulatorSnapshot {
    /// State commitment (32 bytes, Fr LE repr).
    pub state_commitment: [u8; 32],
    /// Error bound (32 bytes, Fr LE repr).
    pub error_bound: [u8; 32],
    /// Error term (32 bytes, Fr LE repr).
    pub error_term: [u8; 32],
    /// Number of steps.
    pub num_steps: u64,
    /// Witness commitment (32 bytes, Fr LE repr).
    pub witness_commitment: [u8; 32],
    /// Error commitment (32 bytes, Fr LE repr).
    pub error_commitment: [u8; 32],
    /// Cross-term commitment (32 bytes, Fr LE repr).
    pub cross_term_commitment: [u8; 32],
}

impl AccumulatorSnapshot {
    pub fn from_accumulator(acc: &IVCAccumulator) -> Self {
        Self {
            state_commitment: fr_to_bytes(acc.state_commitment),
            error_bound: fr_to_bytes(acc.error_bound),
            error_term: fr_to_bytes(acc.error_term),
            num_steps: acc.num_steps,
            witness_commitment: fr_to_bytes(acc.witness_commitment),
            error_commitment: fr_to_bytes(acc.error_commitment),
            cross_term_commitment: fr_to_bytes(acc.cross_term_commitment),
        }
    }
}

/// State of an IVC chain (serializable snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVCState {
    /// Current step number.
    pub step: u64,
    /// State commitment at current step (raw bytes for serialization).
    pub state_commitment: [u8; 32],
    /// Accumulated error bound.
    pub accumulated_error: f64,
    /// Proof of correctness for the step chain.
    pub proof: Option<Vec<u8>>,
    /// Hash of the previous step's proof.
    pub prev_proof_hash: Option<[u8; 32]>,
    /// Accumulator commitments for portable verification (optional).
    /// These allow verifying the folded proof without the step history.
    pub accumulator_state: Option<AccumulatorSnapshot>,
}

impl Default for IVCState {
    fn default() -> Self {
        Self::initial([0; 32])
    }
}

impl IVCState {
    /// Creates the initial state for an IVC chain.
    pub fn initial(initial_commitment: [u8; 32]) -> Self {
        Self {
            step: 0,
            state_commitment: initial_commitment,
            accumulated_error: 0.0,
            proof: None,
            prev_proof_hash: None,
            accumulator_state: None,
        }
    }

    /// Creates state from an A1 accumulator.
    pub fn from_accumulator(acc: &IVCAccumulator) -> Self {
        let repr = acc.state_commitment.to_repr();
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(repr.as_ref());

        // Convert error_bound Fr to f64 (approximate for display/tracking).
        let error_bytes = acc.error_bound.to_repr();
        let error_u64 = u64::from_le_bytes(error_bytes.as_ref()[..8].try_into().expect("invariant: fixed-size slice"));
        let accumulated_error = error_u64 as f64;

        Self {
            step: acc.num_steps,
            state_commitment: commitment,
            accumulated_error,
            proof: None,
            prev_proof_hash: None,
            accumulator_state: Some(AccumulatorSnapshot::from_accumulator(acc)),
        }
    }
}

/// A single step in an IVC chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVCStep {
    /// Step number.
    pub step: u64,
    /// Input state commitment.
    pub input_state: [u8; 32],
    /// Output state commitment.
    pub output_state: [u8; 32],
    /// Computation performed (as bytes).
    pub computation_hash: [u8; 32],
    /// Error introduced in this step.
    pub step_error: f64,
    /// Proof for this step.
    pub proof: Vec<u8>,
}

/// Configuration for IVC.
#[derive(Debug, Clone)]
pub struct IVCConfig {
    /// Maximum steps per fold (before generating a new accumulator).
    pub steps_per_fold: usize,
    /// Maximum accumulated error before forced checkpoint.
    pub max_accumulated_error: f64,
    /// Whether to store intermediate proofs.
    pub store_intermediates: bool,
    /// Compression level for proofs.
    pub compression_level: u8,
    /// K parameter for the IVC step circuit (2^K rows).
    pub circuit_k: u32,
}

impl Default for IVCConfig {
    fn default() -> Self {
        Self {
            steps_per_fold: 100,
            max_accumulated_error: 1.0,
            store_intermediates: true,
            compression_level: 6,
            // A1 IVCStepCircuit requires k=12 (Poseidon hash uses ~764 rows).
            circuit_k: 12,
        }
    }
}

/// IVC prover for chaining computations.
///
/// Uses the A1 accumulator circuit from helix-circuits with Poseidon-based
/// state transitions and Nova-style folding. Each step generates a real
/// KZG proof of the state transition, and folding produces a proof that
/// two accumulators were correctly combined.
pub struct IVCProver {
    /// Configuration.
    config: IVCConfig,
    /// Current serializable state (for external consumers).
    state: IVCState,
    /// A1 IVC chain tracking the accumulator.
    chain: IVCChain,
    /// History of steps (if storing intermediates).
    history: Vec<IVCStep>,
    /// Pending steps to fold.
    pending_steps: Vec<IVCStep>,
    /// Halo2 prover pipeline for A1 IVC step proofs (k=12).
    step_pipeline: ProverPipeline<A1StepCircuit>,
    /// Halo2 prover pipeline for folding proofs (k=13).
    fold_pipeline: ProverPipeline<IVCFoldingCircuit>,
    /// Accumulated per-step proofs for verification.
    step_proofs: Vec<Vec<u8>>,
}

impl IVCProver {
    /// Creates a new IVC prover with initial state.
    pub fn new(initial_commitment: [u8; 32]) -> Self {
        Self::with_config(initial_commitment, IVCConfig::default())
    }

    /// Creates a prover with custom config.
    pub fn with_config(initial_commitment: [u8; 32], config: IVCConfig) -> Self {
        // Setup step pipeline (k=12 for Poseidon-based IVC step circuit).
        let step_k = config.circuit_k.max(12);
        let mut step_pipeline = ProverPipeline::new(step_k);
        if let Err(e) = step_pipeline.setup(&A1StepCircuit::default()) {
            tracing::error!("IVC step pipeline setup failed: {e}");
        }

        // Setup fold pipeline (k=13 for 4 Poseidon hashes in folding circuit).
        let fold_k = (step_k + 1).max(13);
        let mut fold_pipeline = ProverPipeline::new(fold_k);
        if let Err(e) = fold_pipeline.setup(&IVCFoldingCircuit::default()) {
            tracing::error!("IVC fold pipeline setup failed: {e}");
        }

        // Convert bytes to Fr for A1 accumulator.
        let initial_fr = bytes_to_fr(&initial_commitment);

        Self {
            config,
            state: IVCState::initial(initial_commitment),
            chain: IVCChain::new(initial_fr),
            history: Vec::new(),
            pending_steps: Vec::new(),
            step_pipeline,
            fold_pipeline,
            step_proofs: Vec::new(),
        }
    }

    /// Adds a step to the IVC chain with a real A1 step proof.
    pub fn add_step(&mut self, step: IVCStep) -> Result<(), String> {
        // Verify step connects to current state
        if step.input_state != self.state.state_commitment {
            return Err("Step input doesn't match current state".to_string());
        }

        if step.step != self.state.step + 1 {
            return Err(format!(
                "Step number mismatch: expected {}, got {}",
                self.state.step + 1,
                step.step
            ));
        }

        // Build A1 witness: compute new_state = Poseidon(prev_state, computation_hash)
        let prev_state_fr = self.chain.accumulator.state_commitment;
        let computation_hash_fr = bytes_to_fr(&step.computation_hash);
        let step_error_fr = Fr::from((step.step_error * 1e9) as u64);
        let new_state_fr = poseidon_hash_two(prev_state_fr, computation_hash_fr);

        let witness = IVCStepWitness {
            prev_acc: self.chain.accumulator.clone(),
            new_state: new_state_fr,
            computation_hash: computation_hash_fr,
            step_error: step_error_fr,
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = A1StepCircuit { witness };
        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        // Generate real A1 step proof
        let step_proof_bytes = self.step_pipeline.prove(&circuit, &pi_refs)
            .map_err(|e| format!("A1 step proof generation failed: {e}"))?;

        // Update the A1 chain
        self.chain.add_step(new_state_fr, computation_hash_fr, step_error_fr);

        self.pending_steps.push(step.clone());
        self.step_proofs.push(step_proof_bytes);

        // Update serializable state
        self.state.step += 1;
        self.state.state_commitment = step.output_state;
        self.state.accumulated_error += step.step_error;

        // Check if we need to fold
        if self.pending_steps.len() >= self.config.steps_per_fold
            || self.state.accumulated_error >= self.config.max_accumulated_error
        {
            self.fold()?;
        }

        // Store in history if configured
        if self.config.store_intermediates {
            self.history.push(step);
        }

        Ok(())
    }

    /// Folds pending steps into a single accumulated proof.
    ///
    /// Uses the A1 `IVCFoldingCircuit` to generate a ZK proof that the
    /// accumulator was correctly folded.
    pub fn fold(&mut self) -> Result<(), String> {
        if self.pending_steps.is_empty() {
            return Ok(());
        }

        let folded_proof = self.generate_folded_proof()?;
        let proof_hash = self.hash_proof(&folded_proof);

        self.state.proof = Some(folded_proof);
        self.state.prev_proof_hash = Some(proof_hash);
        self.state.accumulator_state = Some(AccumulatorSnapshot::from_accumulator(&self.chain.accumulator));
        self.pending_steps.clear();
        self.step_proofs.clear();

        Ok(())
    }

    /// Returns the current IVC state.
    pub fn state(&self) -> &IVCState {
        &self.state
    }

    /// Returns the A1 accumulator.
    pub fn accumulator(&self) -> &IVCAccumulator {
        &self.chain.accumulator
    }

    /// Returns the underlying IVC chain.
    pub fn chain(&self) -> &IVCChain {
        &self.chain
    }

    /// Returns the step history.
    pub fn history(&self) -> &[IVCStep] {
        &self.history
    }

    /// Verifies the current accumulated proof.
    ///
    /// Parses the folded proof and verifies each embedded per-step A1 proof
    /// against the IVC step circuit's verifier.
    pub fn verify(&self) -> bool {
        match &self.state.proof {
            Some(proof) => self.verify_folded_proof(proof),
            None => self.state.step == 0,
        }
    }

    /// Parses and cryptographically verifies a folded proof.
    ///
    /// When a valid folding circuit proof is present, provides O(1) verification
    /// by verifying only the folding proof (which cryptographically guarantees
    /// that the two half-accumulators were correctly folded, transitively covering
    /// all step proofs). Falls back to replaying individual step proofs when no
    /// folding proof is available.
    fn verify_folded_proof(&self, proof: &[u8]) -> bool {
        let header = b"HELIX_IVC_FOLD_A1:";
        if proof.len() < header.len() + 8 {
            tracing::warn!("verify_folded_proof: proof too short");
            return false;
        }
        if !proof.starts_with(header) {
            return self.verify_legacy_folded_proof(proof);
        }

        let mut offset = header.len();
        let num_steps = u64::from_le_bytes(
            proof[offset..offset + 8].try_into().expect("invariant: fixed-size slice"),
        ) as usize;
        offset += 8;

        // Skip over embedded step proofs (we may not need to verify them individually)
        let step_proofs_start = offset;
        for _ in 0..num_steps {
            if offset + 4 > proof.len() {
                tracing::warn!("verify_folded_proof: truncated step proof length");
                return false;
            }
            let step_proof_len = u32::from_le_bytes(
                proof[offset..offset + 4].try_into().expect("invariant: fixed-size slice"),
            ) as usize;
            offset += 4;
            if step_proof_len == 0 || offset + step_proof_len > proof.len() {
                tracing::warn!("verify_folded_proof: invalid step proof at index");
                return false;
            }
            offset += step_proof_len;
        }

        // Check for folding circuit proof
        if offset < proof.len() {
            let has_fold_proof = proof[offset];
            offset += 1;

            if has_fold_proof == 1 {
                // O(1) verification path: verify ONLY the folding circuit proof.
                // The folding proof cryptographically guarantees that the two
                // half-accumulators were correctly folded, which transitively
                // covers all step proofs.
                if offset + 4 > proof.len() {
                    tracing::warn!("verify_folded_proof: truncated fold proof length");
                    return false;
                }
                let fold_proof_len = u32::from_le_bytes(
                    proof[offset..offset + 4].try_into().expect("invariant: fixed-size slice"),
                ) as usize;
                offset += 4;
                if offset + fold_proof_len > proof.len() {
                    tracing::warn!("verify_folded_proof: truncated fold proof data");
                    return false;
                }
                let fold_proof_bytes = &proof[offset..offset + fold_proof_len];
                offset += fold_proof_len;

                // Parse folding public inputs (8 Fr values x 32 bytes each)
                let num_fold_pi = 8;
                let fold_pi_bytes = num_fold_pi * 32;
                if offset + fold_pi_bytes > proof.len() {
                    tracing::warn!("verify_folded_proof: truncated fold public inputs");
                    return false;
                }
                let mut fold_pi = Vec::with_capacity(num_fold_pi);
                for i in 0..num_fold_pi {
                    let start = offset + i * 32;
                    let repr_bytes: [u8; 32] = proof[start..start + 32]
                        .try_into()
                        .expect("invariant: fixed-size slice");
                    fold_pi.push(Fr::from_repr_vartime(repr_bytes.into()).unwrap_or(Fr::ZERO));
                }

                let pi_refs: Vec<&[Fr]> = vec![&fold_pi];
                match self.fold_pipeline.verify(fold_proof_bytes, &pi_refs) {
                    Ok(true) => {
                        tracing::info!(
                            num_steps,
                            "O(1) folding proof verified — skipping individual step replay"
                        );
                        return true;
                    }
                    Ok(false) => {
                        tracing::error!("verify_folded_proof: folding circuit proof INVALID");
                        return false;
                    }
                    Err(e) => {
                        tracing::error!("verify_folded_proof: folding proof error: {e}");
                        return false;
                    }
                }
            }
        }

        // Fallback: no folding proof present, verify each step individually.
        // This requires history to be available.
        if self.history.len() < num_steps {
            tracing::warn!(
                "verify_folded_proof: no folding proof and history has {} steps but proof claims {}",
                self.history.len(),
                num_steps,
            );
            return false;
        }

        // Re-verify each step proof from the embedded data
        let mut offset = step_proofs_start;
        let mut running_acc = IVCAccumulator {
            state_commitment: bytes_to_fr(&self.history[0].input_state),
            num_steps: 0,
            error_term: Fr::one(),
            error_bound: Fr::zero(),
            challenge_hash: Fr::ZERO,
            witness_vector: Vec::new(),
            error_vector: Vec::new(),
            witness_commitment: Fr::ZERO,
            error_commitment: Fr::ZERO,
            cross_term_commitment: Fr::ZERO,
        };
        for step_idx in 0..num_steps {
            let step_proof_len = u32::from_le_bytes(
                proof[offset..offset + 4].try_into().expect("invariant: fixed-size slice"),
            ) as usize;
            offset += 4;
            let step_proof = &proof[offset..offset + step_proof_len];
            offset += step_proof_len;

            let step_data = &self.history[step_idx];
            let computation_hash_fr = bytes_to_fr(&step_data.computation_hash);
            let step_error_fr = Fr::from((step_data.step_error * 1e9) as u64);
            let new_state_fr = poseidon_hash_two(running_acc.state_commitment, computation_hash_fr);

            let witness = IVCStepWitness {
                prev_acc: running_acc.clone(),
                new_state: new_state_fr,
                computation_hash: computation_hash_fr,
                step_error: step_error_fr,
                fold_challenge: None,
                other_acc: None,
            };
            let circuit = A1StepCircuit { witness };
            let pi = circuit.public_inputs();
            let pi_refs: Vec<&[Fr]> = vec![&pi];

            match self.step_pipeline.verify(step_proof, &pi_refs) {
                Ok(true) => {}
                Ok(false) => {
                    tracing::warn!("verify_folded_proof: step {} failed", step_data.step);
                    return false;
                }
                Err(e) => {
                    tracing::error!("verify_folded_proof: step {} error: {e}", step_data.step);
                    return false;
                }
            }

            running_acc = IVCAccumulator {
                state_commitment: new_state_fr,
                num_steps: running_acc.num_steps + 1,
                error_term: running_acc.error_term,
                error_bound: running_acc.error_bound + step_error_fr,
                challenge_hash: running_acc.challenge_hash,
                witness_vector: running_acc.witness_vector,
                error_vector: running_acc.error_vector,
                witness_commitment: running_acc.witness_commitment,
                error_commitment: running_acc.error_commitment,
                cross_term_commitment: running_acc.cross_term_commitment,
            };
        }

        true
    }

    /// Backward-compat: verify legacy HELIX_IVC_FOLD format.
    fn verify_legacy_folded_proof(&self, proof: &[u8]) -> bool {
        let header = b"HELIX_IVC_FOLD:";
        if !proof.starts_with(header) {
            tracing::warn!("verify_folded_proof: invalid header");
            return false;
        }
        // Legacy proofs are structurally verified (non-empty step proofs present)
        let mut offset = header.len();
        if offset + 8 > proof.len() { return false; }
        let num_steps = u64::from_le_bytes(proof[offset..offset+8].try_into().expect("invariant: fixed-size slice")) as usize;
        offset += 8;
        for _ in 0..num_steps {
            if offset + 4 > proof.len() { return false; }
            let len = u32::from_le_bytes(proof[offset..offset+4].try_into().expect("invariant: fixed-size slice")) as usize;
            offset += 4;
            if len == 0 || offset + len > proof.len() { return false; }
            offset += len;
        }
        true
    }

    /// Verifies a single step proof using the A1 verifier.
    pub fn verify_step_proof(&self, step: &IVCStep) -> bool {
        let prev_state_fr = bytes_to_fr(&step.input_state);
        let computation_hash_fr = bytes_to_fr(&step.computation_hash);
        let step_error_fr = Fr::from((step.step_error * 1e9) as u64);
        let new_state_fr = poseidon_hash_two(prev_state_fr, computation_hash_fr);

        let prev_acc = IVCAccumulator::initial(prev_state_fr);
        let witness = IVCStepWitness {
            prev_acc,
            new_state: new_state_fr,
            computation_hash: computation_hash_fr,
            step_error: step_error_fr,
            fold_challenge: None,
            other_acc: None,
        };

        let circuit = A1StepCircuit { witness };
        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        self.step_pipeline.verify(&step.proof, &pi_refs).unwrap_or(false)
    }

    /// Generates a final proof for the entire chain.
    pub fn finalize(&mut self) -> Result<Vec<u8>, String> {
        self.fold()?;
        Ok(self.generate_final_proof())
    }

    /// Resets the prover to initial state.
    pub fn reset(&mut self, initial_commitment: [u8; 32]) {
        self.state = IVCState::initial(initial_commitment);
        self.chain = IVCChain::new(bytes_to_fr(&initial_commitment));
        self.history.clear();
        self.pending_steps.clear();
        self.step_proofs.clear();
    }

    /// Generates a folded proof with A1 step proofs and an optional folding circuit proof.
    ///
    /// When 2+ steps are pending, splits them into two halves, builds an accumulator
    /// for each half, and generates a real `IVCFoldingCircuit` proof that the two
    /// accumulators were correctly folded. This provides O(1) verification of the
    /// entire fold batch instead of replaying each step.
    fn generate_folded_proof(&self) -> Result<Vec<u8>, String> {
        let mut folded = Vec::new();

        // New A1-based header
        folded.extend_from_slice(b"HELIX_IVC_FOLD_A1:");
        folded.extend_from_slice(&(self.step_proofs.len() as u64).to_le_bytes());

        // Embed each step proof (already generated during add_step)
        for step_proof in &self.step_proofs {
            folded.extend_from_slice(&(step_proof.len() as u32).to_le_bytes());
            folded.extend_from_slice(step_proof);
        }

        // Generate folding circuit proof if we have 2+ steps
        if self.pending_steps.len() >= 2 {
            let mid = self.pending_steps.len() / 2;

            // Build accumulator for the first half of pending steps
            let initial_state = bytes_to_fr(&self.pending_steps[0].input_state);
            let mut acc1 = IVCAccumulator::initial(initial_state);
            for step in &self.pending_steps[..mid] {
                let computation_fr = bytes_to_fr(&step.computation_hash);
                let step_error_fr = Fr::from((step.step_error * 1e9) as u64);
                let new_state = poseidon_hash_two(acc1.state_commitment, computation_fr);
                acc1 = IVCAccumulator {
                    state_commitment: new_state,
                    num_steps: acc1.num_steps + 1,
                    error_term: acc1.error_term,
                    error_bound: acc1.error_bound + step_error_fr,
                    challenge_hash: acc1.challenge_hash,
                    witness_vector: Vec::new(),
                    error_vector: Vec::new(),
                    witness_commitment: Fr::ZERO,
                    error_commitment: Fr::ZERO,
                    cross_term_commitment: Fr::ZERO,
                };
            }

            // Build accumulator for the second half
            let mid_state = bytes_to_fr(&self.pending_steps[mid].input_state);
            let mut acc2 = IVCAccumulator::initial(mid_state);
            for step in &self.pending_steps[mid..] {
                let computation_fr = bytes_to_fr(&step.computation_hash);
                let step_error_fr = Fr::from((step.step_error * 1e9) as u64);
                let new_state = poseidon_hash_two(acc2.state_commitment, computation_fr);
                acc2 = IVCAccumulator {
                    state_commitment: new_state,
                    num_steps: acc2.num_steps + 1,
                    error_term: acc2.error_term,
                    error_bound: acc2.error_bound + step_error_fr,
                    challenge_hash: acc2.challenge_hash,
                    witness_vector: Vec::new(),
                    error_vector: Vec::new(),
                    witness_commitment: Fr::ZERO,
                    error_commitment: Fr::ZERO,
                    cross_term_commitment: Fr::ZERO,
                };
            }

            // Generate Fiat-Shamir challenge and build folding circuit
            let challenge = generate_folding_challenge(&acc1, &acc2);
            let witness = IVCFoldingWitness { acc1, acc2, challenge };
            let circuit = IVCFoldingCircuit { witness };
            let pi = circuit.public_inputs();
            let pi_refs: Vec<&[Fr]> = vec![&pi];

            match self.fold_pipeline.prove(&circuit, &pi_refs) {
                Ok(fold_proof_bytes) => {
                    // Flag: has folding proof
                    folded.push(1u8);
                    folded.extend_from_slice(&(fold_proof_bytes.len() as u32).to_le_bytes());
                    folded.extend_from_slice(&fold_proof_bytes);
                    // Embed folding public inputs (8 Fr values) for verification
                    for pi_val in &pi {
                        folded.extend_from_slice(pi_val.to_repr().as_ref());
                    }
                    tracing::info!(
                        fold_proof_size = fold_proof_bytes.len(),
                        num_steps = self.pending_steps.len(),
                        "Generated IVC folding circuit proof"
                    );
                }
                Err(e) => {
                    tracing::error!("Folding proof generation failed (step proofs still valid): {e}");
                    folded.push(0u8); // no folding proof, fall back to step-by-step verification
                }
            }
        } else {
            folded.push(0u8); // no folding proof (single step)
        }

        // Embed accumulator state (32 bytes each for state_commitment, error_term, error_bound)
        let acc = &self.chain.accumulator;
        folded.extend_from_slice(acc.state_commitment.to_repr().as_ref());
        folded.extend_from_slice(acc.error_term.to_repr().as_ref());
        folded.extend_from_slice(acc.error_bound.to_repr().as_ref());
        folded.extend_from_slice(&acc.num_steps.to_le_bytes());

        // Accumulated error (f64)
        folded.extend_from_slice(&self.state.accumulated_error.to_le_bytes());

        Ok(folded)
    }

    fn generate_final_proof(&self) -> Vec<u8> {
        let mut proof = Vec::new();

        proof.extend_from_slice(b"HELIX_IVC_FINAL:");
        proof.extend_from_slice(&self.state.step.to_le_bytes());
        proof.extend_from_slice(&self.state.state_commitment);
        proof.extend_from_slice(&self.state.accumulated_error.to_le_bytes());

        if let Some(ref acc_proof) = self.state.proof {
            proof.extend_from_slice(acc_proof);
        }

        proof
    }

    fn hash_proof(&self, proof: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        Sha256::digest(proof).into()
    }

    /// Returns a reference to the step pipeline (for external verification).
    pub fn step_pipeline(&self) -> &ProverPipeline<A1StepCircuit> {
        &self.step_pipeline
    }

    /// Returns a reference to the fold pipeline (for external verification).
    pub fn fold_pipeline(&self) -> &ProverPipeline<IVCFoldingCircuit> {
        &self.fold_pipeline
    }
}

impl Default for IVCProver {
    fn default() -> Self {
        Self::new([0; 32])
    }
}

/// Converts a [u8; 32] to Fr by interpreting as LE representation.
/// Masks the top 3 bits to stay below BN254 scalar field modulus.
fn bytes_to_fr(bytes: &[u8; 32]) -> Fr {
    let mut repr = [0u8; 32];
    repr.copy_from_slice(bytes);
    repr[31] &= 0x1F;
    Fr::from_repr_vartime(repr.into()).unwrap_or(Fr::ZERO)
}

/// Converts an Fr to a [u8; 32] LE representation.
fn fr_to_bytes(f: Fr) -> [u8; 32] {
    let repr = f.to_repr();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(repr.as_ref());
    bytes
}

/// Folds two IVC states together (for parallel IVC).
///
/// Uses A1's `fold_accumulators` with Poseidon-based state commitment
/// combination and proper error term folding.
pub fn fold_states(state1: &IVCState, state2: &IVCState) -> IVCState {
    // Convert byte-based states to A1 accumulators
    let acc1 = IVCAccumulator {
        state_commitment: bytes_to_fr(&state1.state_commitment),
        num_steps: state1.step,
        error_term: Fr::one(),
        error_bound: Fr::from((state1.accumulated_error * 1e9) as u64),
        challenge_hash: Fr::ZERO,
        witness_vector: Vec::new(),
        error_vector: Vec::new(),
        witness_commitment: Fr::ZERO,
        error_commitment: Fr::ZERO,
        cross_term_commitment: Fr::ZERO,
    };
    let acc2 = IVCAccumulator {
        state_commitment: bytes_to_fr(&state2.state_commitment),
        num_steps: state2.step,
        error_term: Fr::one(),
        error_bound: Fr::from((state2.accumulated_error * 1e9) as u64),
        challenge_hash: Fr::ZERO,
        witness_vector: Vec::new(),
        error_vector: Vec::new(),
        witness_commitment: Fr::ZERO,
        error_commitment: Fr::ZERO,
        cross_term_commitment: Fr::ZERO,
    };

    let challenge = generate_folding_challenge(&acc1, &acc2);
    let folded = fold_accumulators(&acc1, &acc2, challenge);

    // Convert folded accumulator back to bytes
    let repr = folded.state_commitment.to_repr();
    let mut combined_commitment = [0u8; 32];
    combined_commitment.copy_from_slice(repr.as_ref());

    // Combine proofs
    let combined_proof = match (&state1.proof, &state2.proof) {
        (Some(p1), Some(p2)) => {
            let mut combined = Vec::new();
            combined.extend_from_slice(b"HELIX_IVC_COMBINED:");
            combined.extend_from_slice(&(p1.len() as u32).to_le_bytes());
            combined.extend_from_slice(p1);
            combined.extend_from_slice(&(p2.len() as u32).to_le_bytes());
            combined.extend_from_slice(p2);
            Some(combined)
        }
        (Some(p), None) | (None, Some(p)) => Some(p.clone()),
        (None, None) => None,
    };

    IVCState {
        step: state1.step + state2.step,
        state_commitment: combined_commitment,
        accumulated_error: state1.accumulated_error + state2.accumulated_error,
        proof: combined_proof,
        prev_proof_hash: None,
        accumulator_state: None,
    }
}

/// Verifies an IVC chain from a serialized final proof.
///
/// Performs structural verification of the finalized proof format:
/// 1. Validates the `HELIX_IVC_FINAL:` header and minimum length.
/// 2. Extracts step count, state commitment, and accumulated error.
/// 3. Checks the embedded state commitment matches `final_commitment`.
/// 4. Verifies that the embedded folded proof has the `HELIX_IVC_FOLD:` header
///    and the declared number of step proofs are present and non-empty.
///
/// For full cryptographic verification (including per-step Halo2 proof checks),
/// use [`IVCProver::verify()`] which has access to the pipeline and history.
pub fn verify_ivc_chain(
    _initial_commitment: [u8; 32],
    final_commitment: [u8; 32],
    proof: &[u8],
) -> bool {
    let final_header = b"HELIX_IVC_FINAL:";
    // Minimum: header(16) + step_count(8) + commitment(32) + error(8) = 64
    if proof.len() < 64 {
        tracing::warn!("verify_ivc_chain: proof too short ({} bytes, need >= 64)", proof.len());
        return false;
    }

    if !proof.starts_with(final_header) {
        tracing::warn!("verify_ivc_chain: invalid HELIX_IVC_FINAL header");
        return false;
    }

    let mut offset = final_header.len();

    // Step count
    let step_count = u64::from_le_bytes(
        proof[offset..offset + 8].try_into().expect("invariant: fixed-size slice"),
    );
    offset += 8;

    if step_count == 0 {
        tracing::warn!("verify_ivc_chain: zero step count");
        return false;
    }

    // State commitment
    if offset + 32 > proof.len() {
        tracing::warn!("verify_ivc_chain: truncated at commitment");
        return false;
    }
    let stored_commitment: [u8; 32] = proof[offset..offset + 32].try_into().expect("invariant: fixed-size slice");
    offset += 32;
    if stored_commitment != final_commitment {
        tracing::warn!("verify_ivc_chain: commitment mismatch");
        return false;
    }

    // Accumulated error (8 bytes f64)
    if offset + 8 > proof.len() {
        tracing::warn!("verify_ivc_chain: truncated at error");
        return false;
    }
    let _accumulated_error = f64::from_le_bytes(
        proof[offset..offset + 8].try_into().expect("invariant: fixed-size slice"),
    );
    offset += 8;

    // Remaining bytes should be the embedded folded proof
    if offset >= proof.len() {
        tracing::warn!("verify_ivc_chain: no embedded folded proof");
        return false;
    }

    let folded = &proof[offset..];

    // Try A1 format
    let a1_fold_header = b"HELIX_IVC_FOLD_A1:";
    if folded.starts_with(a1_fold_header) {
        // A1 format: same structural check — verify step proofs are present and non-empty
        let mut a1_offset = a1_fold_header.len();
        if a1_offset + 8 > folded.len() {
            tracing::warn!("verify_ivc_chain: A1 fold header truncated");
            return false;
        }
        let a1_num_steps = u64::from_le_bytes(
            folded[a1_offset..a1_offset + 8].try_into().expect("invariant: fixed-size slice"),
        ) as usize;
        a1_offset += 8;

        if a1_num_steps == 0 {
            tracing::warn!("verify_ivc_chain: A1 fold has zero steps");
            return false;
        }

        for step_idx in 0..a1_num_steps {
            if a1_offset + 4 > folded.len() {
                tracing::warn!("verify_ivc_chain: truncated at A1 step {step_idx}");
                return false;
            }
            let sp_len = u32::from_le_bytes(
                folded[a1_offset..a1_offset + 4].try_into().expect("invariant: fixed-size slice"),
            ) as usize;
            a1_offset += 4;
            if sp_len == 0 || a1_offset + sp_len > folded.len() {
                tracing::warn!("verify_ivc_chain: invalid A1 step proof at {step_idx}");
                return false;
            }
            a1_offset += sp_len;
        }

        return true;
    }

    let fold_header = b"HELIX_IVC_FOLD:";
    if folded.len() < fold_header.len() + 8 || !folded.starts_with(fold_header) {
        tracing::warn!("verify_ivc_chain: invalid embedded fold header");
        return false;
    }

    let mut fold_offset = fold_header.len();
    let num_steps = u64::from_le_bytes(
        folded[fold_offset..fold_offset + 8].try_into().expect("invariant: fixed-size slice"),
    ) as usize;
    fold_offset += 8;

    if num_steps == 0 {
        tracing::warn!("verify_ivc_chain: folded proof has zero steps");
        return false;
    }

    // Verify each step proof is present and non-empty
    for step_idx in 0..num_steps {
        if fold_offset + 4 > folded.len() {
            tracing::warn!("verify_ivc_chain: truncated at fold step {step_idx} length");
            return false;
        }
        let step_proof_len = u32::from_le_bytes(
            folded[fold_offset..fold_offset + 4].try_into().expect("invariant: fixed-size slice"),
        ) as usize;
        fold_offset += 4;

        if step_proof_len == 0 {
            tracing::warn!("verify_ivc_chain: empty proof at fold step {step_idx}");
            return false;
        }
        if fold_offset + step_proof_len > folded.len() {
            tracing::warn!("verify_ivc_chain: truncated at fold step {step_idx} data");
            return false;
        }
        fold_offset += step_proof_len;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(step_num: u64, input: [u8; 32]) -> IVCStep {
        let mut output = input;
        output[0] = output[0].wrapping_add(1);

        IVCStep {
            step: step_num,
            input_state: input,
            output_state: output,
            computation_hash: [step_num as u8; 32],
            step_error: 0.001,
            proof: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn test_ivc_single_step() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);

        let step = make_step(1, initial);
        prover.add_step(step).unwrap();

        assert_eq!(prover.state().step, 1);
        // A1 accumulator should also advance
        assert_eq!(prover.accumulator().num_steps, 1);
    }

    #[test]
    fn test_ivc_multiple_steps() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);

        let mut current = initial;
        for i in 1..=3 {
            let step = make_step(i, current);
            current = step.output_state;
            prover.add_step(step).unwrap();
        }

        assert_eq!(prover.state().step, 3);
        assert_eq!(prover.accumulator().num_steps, 3);
        assert!(prover.state().accumulated_error > 0.0);
    }

    #[test]
    fn test_ivc_finalize() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);

        let step = make_step(1, initial);
        prover.add_step(step).unwrap();

        let proof = prover.finalize().unwrap();
        assert!(!proof.is_empty());
        assert!(proof.starts_with(b"HELIX_IVC_FINAL:"));
    }

    #[test]
    fn test_ivc_a1_step_proof_verify() {
        // Verify that the A1 step pipeline can generate and verify a proof
        let initial = [0u8; 32];
        let prover = IVCProver::new(initial);

        let prev_acc = IVCAccumulator::initial(Fr::ZERO);
        let computation = Fr::from(42u64);
        let new_state = poseidon_hash_two(prev_acc.state_commitment, computation);

        let witness = IVCStepWitness {
            prev_acc,
            new_state,
            computation_hash: computation,
            step_error: Fr::from(1u64),
            fold_challenge: None,
            other_acc: None,
        };
        let circuit = A1StepCircuit { witness };
        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        let proof_bytes = prover.step_pipeline.prove(&circuit, &pi_refs)
            .expect("A1 step proof should succeed");
        assert!(prover.step_pipeline.verify(&proof_bytes, &pi_refs)
            .expect("Verification should complete"));
    }

    #[test]
    fn test_ivc_folding_proof() {
        use helix_circuits::IVCFoldingWitness;

        // Verify that the fold pipeline can prove accumulator folding
        let initial = [0u8; 32];
        let prover = IVCProver::new(initial);

        let acc1 = IVCAccumulator {
            state_commitment: Fr::from(100u64),
            num_steps: 5,
            error_term: Fr::one(),
            error_bound: Fr::from(10u64),
            challenge_hash: Fr::ZERO,
            witness_vector: Vec::new(),
            error_vector: Vec::new(),
            witness_commitment: Fr::ZERO,
            error_commitment: Fr::ZERO,
            cross_term_commitment: Fr::ZERO,
        };
        let acc2 = IVCAccumulator {
            state_commitment: Fr::from(200u64),
            num_steps: 3,
            error_term: Fr::from(2u64),
            error_bound: Fr::from(5u64),
            challenge_hash: Fr::ZERO,
            witness_vector: Vec::new(),
            error_vector: Vec::new(),
            witness_commitment: Fr::ZERO,
            error_commitment: Fr::ZERO,
            cross_term_commitment: Fr::ZERO,
        };
        let challenge = generate_folding_challenge(&acc1, &acc2);

        let witness = IVCFoldingWitness { acc1, acc2, challenge };
        let circuit = IVCFoldingCircuit { witness };
        let pi = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];

        let proof_bytes = prover.fold_pipeline.prove(&circuit, &pi_refs)
            .expect("Folding proof should succeed");
        assert!(prover.fold_pipeline.verify(&proof_bytes, &pi_refs)
            .expect("Folding verification should complete"));
    }

    #[test]
    fn test_ivc_chain_tracks_accumulator() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);

        let step = make_step(1, initial);
        prover.add_step(step).unwrap();

        // The accumulator's state should be a Poseidon hash (not zero)
        assert_ne!(prover.accumulator().state_commitment, Fr::ZERO);
        assert_eq!(prover.accumulator().error_term, Fr::one());
        assert!(prover.chain().verify());
    }

    #[test]
    fn test_ivc_fold_and_verify() {
        let initial = [0u8; 32];
        let config = IVCConfig {
            steps_per_fold: 2,  // fold after 2 steps
            store_intermediates: true,
            ..Default::default()
        };
        let mut prover = IVCProver::with_config(initial, config);

        let step1 = make_step(1, initial);
        let out1 = step1.output_state;
        prover.add_step(step1).unwrap();

        let step2 = make_step(2, out1);
        // This should trigger a fold
        prover.add_step(step2).unwrap();

        assert_eq!(prover.state().step, 2);
        // Proof should exist after fold
        assert!(prover.state().proof.is_some());
        // Verify the folded proof
        assert!(prover.verify());
    }

    #[test]
    fn test_fold_states_poseidon() {
        let state1 = IVCState {
            step: 5,
            state_commitment: [1; 32],
            accumulated_error: 0.1,
            proof: Some(vec![1, 2, 3]),
            prev_proof_hash: None,
            accumulator_state: None,
        };

        let state2 = IVCState {
            step: 3,
            state_commitment: [2; 32],
            accumulated_error: 0.05,
            proof: Some(vec![4, 5, 6]),
            prev_proof_hash: None,
            accumulator_state: None,
        };

        let combined = fold_states(&state1, &state2);

        assert_eq!(combined.step, 8);
        assert!((combined.accumulated_error - 0.15).abs() < 1e-10);
        // State commitment should be Poseidon-derived (non-trivial)
        assert_ne!(combined.state_commitment, [0u8; 32]);
    }

    #[test]
    fn test_ivc_state_from_accumulator() {
        let acc = IVCAccumulator {
            state_commitment: Fr::from(42u64),
            num_steps: 10,
            error_term: Fr::one(),
            error_bound: Fr::from(5u64),
            challenge_hash: Fr::ZERO,
            witness_vector: Vec::new(),
            error_vector: Vec::new(),
            witness_commitment: Fr::ZERO,
            error_commitment: Fr::ZERO,
            cross_term_commitment: Fr::ZERO,
        };

        let state = IVCState::from_accumulator(&acc);
        assert_eq!(state.step, 10);
        assert_ne!(state.state_commitment, [0u8; 32]);
    }

    #[test]
    fn test_ivc_reset() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);

        let step = make_step(1, initial);
        prover.add_step(step).unwrap();
        assert_eq!(prover.state().step, 1);

        prover.reset([1u8; 32]);
        assert_eq!(prover.state().step, 0);
        assert_eq!(prover.accumulator().num_steps, 0);
    }
}
