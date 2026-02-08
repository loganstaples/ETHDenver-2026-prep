//! Incrementally Verifiable Computation (IVC).
//!
//! Implements IVC for proving long-running ML computations step by step,
//! where each step's proof can be verified and extended.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing;

use crate::pipeline::ProverPipeline;
use crate::provers::ivc_circuit::IVCStepCircuit;

/// State of an IVC chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVCState {
    /// Current step number.
    pub step: u64,
    /// State commitment at current step.
    pub state_commitment: [u8; 32],
    /// Accumulated error bound.
    pub accumulated_error: f64,
    /// Proof of correctness for the step chain.
    pub proof: Option<Vec<u8>>,
    /// Hash of the previous step's proof.
    pub prev_proof_hash: Option<[u8; 32]>,
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
            circuit_k: 5,
        }
    }
}

/// IVC prover for chaining computations.
pub struct IVCProver {
    /// Configuration.
    config: IVCConfig,
    /// Current state.
    state: IVCState,
    /// History of steps (if storing intermediates).
    history: Vec<IVCStep>,
    /// Pending steps to fold.
    pending_steps: Vec<IVCStep>,
    /// Halo2 prover pipeline for IVC step proofs.
    pipeline: ProverPipeline<IVCStepCircuit>,
}

impl IVCProver {
    /// Creates a new IVC prover with initial state.
    pub fn new(initial_commitment: [u8; 32]) -> Self {
        Self::with_config(initial_commitment, IVCConfig::default())
    }

    /// Creates a prover with custom config.
    pub fn with_config(initial_commitment: [u8; 32], config: IVCConfig) -> Self {
        let mut pipeline = ProverPipeline::new(config.circuit_k);
        pipeline.setup(&IVCStepCircuit::default());

        Self {
            config,
            state: IVCState::initial(initial_commitment),
            history: Vec::new(),
            pending_steps: Vec::new(),
            pipeline,
        }
    }

    /// Adds a step to the IVC chain.
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

        self.pending_steps.push(step.clone());

        // Update state
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
    pub fn fold(&mut self) -> Result<(), String> {
        if self.pending_steps.is_empty() {
            return Ok(());
        }

        let folded_proof = self.generate_folded_proof(&self.pending_steps);
        let proof_hash = self.hash_proof(&folded_proof);

        self.state.proof = Some(folded_proof);
        self.state.prev_proof_hash = Some(proof_hash);
        self.pending_steps.clear();

        Ok(())
    }

    /// Returns the current IVC state.
    pub fn state(&self) -> &IVCState {
        &self.state
    }

    /// Returns the step history.
    pub fn history(&self) -> &[IVCStep] {
        &self.history
    }

    /// Verifies the current accumulated proof against the Halo2 pipeline.
    ///
    /// Parses the folded proof (generated by [`fold`]) and verifies each
    /// embedded per-step Halo2 proof against the IVC step circuit's verifier.
    /// Returns `false` if any step proof fails verification or the format
    /// is invalid.
    pub fn verify(&self) -> bool {
        match &self.state.proof {
            Some(proof) => self.verify_folded_proof(proof),
            None => self.state.step == 0, // No proof needed before first step
        }
    }

    /// Parses and cryptographically verifies a folded proof.
    fn verify_folded_proof(&self, proof: &[u8]) -> bool {
        // Format: "HELIX_IVC_FOLD:" [num_steps:8 LE] [step_proofs...] [error:8 LE]
        let header = b"HELIX_IVC_FOLD:";
        if proof.len() < header.len() + 8 {
            tracing::warn!("verify_folded_proof: proof too short");
            return false;
        }
        if !proof.starts_with(header) {
            tracing::warn!("verify_folded_proof: invalid header");
            return false;
        }

        let mut offset = header.len();
        let num_steps = u64::from_le_bytes(
            proof[offset..offset + 8].try_into().unwrap(),
        ) as usize;
        offset += 8;

        // We need the step history to reconstruct expected public inputs
        if self.history.len() < num_steps {
            tracing::warn!(
                "verify_folded_proof: history has {} steps but proof claims {}",
                self.history.len(),
                num_steps,
            );
            return false;
        }

        // Verify each embedded step proof
        for step_idx in 0..num_steps {
            if offset + 4 > proof.len() {
                tracing::warn!("verify_folded_proof: truncated at step {step_idx} length");
                return false;
            }

            let step_proof_len = u32::from_le_bytes(
                proof[offset..offset + 4].try_into().unwrap(),
            ) as usize;
            offset += 4;

            if offset + step_proof_len > proof.len() {
                tracing::warn!(
                    "verify_folded_proof: truncated at step {step_idx} data (need {step_proof_len}, have {})",
                    proof.len() - offset,
                );
                return false;
            }

            let step_proof = &proof[offset..offset + step_proof_len];
            offset += step_proof_len;

            if step_proof.is_empty() {
                tracing::warn!("verify_folded_proof: empty proof at step {step_idx}");
                return false;
            }

            // Reconstruct the circuit for this step and verify the proof
            let step_data = &self.history[step_idx];
            let circuit = IVCStepCircuit {
                prev_state: step_data.input_state,
                new_state: step_data.output_state,
                computation_hash: step_data.computation_hash,
                step_number: step_data.step,
            };

            use helix_circuits::halo2curves::bn256::Fr;
            let pi: Vec<Fr> = circuit.public_inputs();
            let pi_refs: Vec<&[Fr]> = vec![&pi];

            match self.pipeline.verify(step_proof, &pi_refs) {
                Ok(true) => {}
                Ok(false) => {
                    tracing::warn!(
                        "verify_folded_proof: Halo2 verification failed at step {}",
                        step_data.step
                    );
                    return false;
                }
                Err(e) => {
                    tracing::error!(
                        "verify_folded_proof: verification error at step {}: {e}",
                        step_data.step
                    );
                    return false;
                }
            }
        }

        true
    }

    /// Verifies a single step proof using the Halo2 verifier.
    pub fn verify_step_proof(&self, step: &IVCStep) -> bool {
        use helix_circuits::halo2curves::bn256::Fr;

        let circuit = IVCStepCircuit {
            prev_state: step.input_state,
            new_state: step.output_state,
            computation_hash: step.computation_hash,
            step_number: step.step,
        };

        let pi: Vec<Fr> = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        self.pipeline.verify(&step.proof, &pi_refs).unwrap_or(false)
    }

    /// Generates a final proof for the entire chain.
    pub fn finalize(&mut self) -> Result<Vec<u8>, String> {
        // Fold any remaining pending steps
        self.fold()?;

        // Generate final proof
        let final_proof = self.generate_final_proof();
        Ok(final_proof)
    }

    /// Resets the prover to initial state.
    pub fn reset(&mut self, initial_commitment: [u8; 32]) {
        self.state = IVCState::initial(initial_commitment);
        self.history.clear();
        self.pending_steps.clear();
    }

    fn generate_folded_proof(&self, steps: &[IVCStep]) -> Vec<u8> {
        use helix_circuits::halo2curves::bn256::Fr;

        let mut folded = Vec::new();

        // Header
        folded.extend_from_slice(b"HELIX_IVC_FOLD:");
        folded.extend_from_slice(&(steps.len() as u64).to_le_bytes());

        // Generate a real Halo2 proof for each step and embed it.
        for step in steps {
            let circuit = IVCStepCircuit {
                prev_state: step.input_state,
                new_state: step.output_state,
                computation_hash: step.computation_hash,
                step_number: step.step,
            };

            let pi: Vec<Fr> = circuit.public_inputs();
            let pi_refs: Vec<&[Fr]> = vec![&pi];
            let step_proof = self.pipeline.prove(&circuit, &pi_refs)
                .unwrap_or_else(|e| {
                    tracing::error!("IVC folded step proof generation failed at step {}: {e}", step.step);
                    Vec::new()
                });

            // Length-prefixed proof bytes
            folded.extend_from_slice(&(step_proof.len() as u32).to_le_bytes());
            folded.extend_from_slice(&step_proof);
        }

        // Accumulated error
        folded.extend_from_slice(&self.state.accumulated_error.to_le_bytes());

        folded
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
}

impl Default for IVCProver {
    fn default() -> Self {
        Self::new([0; 32])
    }
}

/// Folds two IVC states together (for parallel IVC).
pub fn fold_states(state1: &IVCState, state2: &IVCState) -> IVCState {
    use sha2::{Sha256, Digest};

    // Compute combined commitment.
    let mut hasher = Sha256::new();
    hasher.update(&state1.state_commitment);
    hasher.update(&state2.state_commitment);
    let combined_commitment: [u8; 32] = hasher.finalize().into();

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
        proof[offset..offset + 8].try_into().unwrap(),
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
    let stored_commitment: [u8; 32] = proof[offset..offset + 32].try_into().unwrap();
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
        proof[offset..offset + 8].try_into().unwrap(),
    );
    offset += 8;

    // Remaining bytes should be the embedded folded proof
    if offset >= proof.len() {
        tracing::warn!("verify_ivc_chain: no embedded folded proof");
        return false;
    }

    let folded = &proof[offset..];
    let fold_header = b"HELIX_IVC_FOLD:";
    if folded.len() < fold_header.len() + 8 || !folded.starts_with(fold_header) {
        tracing::warn!("verify_ivc_chain: invalid embedded fold header");
        return false;
    }

    let mut fold_offset = fold_header.len();
    let num_steps = u64::from_le_bytes(
        folded[fold_offset..fold_offset + 8].try_into().unwrap(),
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
            folded[fold_offset..fold_offset + 4].try_into().unwrap(),
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
    }

    #[test]
    fn test_ivc_multiple_steps() {
        let initial = [0u8; 32];
        let mut prover = IVCProver::new(initial);
        
        let mut current = initial;
        for i in 1..=5 {
            let step = make_step(i, current);
            current = step.output_state;
            prover.add_step(step).unwrap();
        }
        
        assert_eq!(prover.state().step, 5);
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
    }

    #[test]
    fn test_ivc_real_proof_verify() {
        use helix_circuits::halo2curves::bn256::Fr;

        let initial = [0u8; 32];
        let prover = IVCProver::new(initial);

        // Build a step and generate a real proof for it via the pipeline
        let mut output = initial;
        output[0] = 1;
        let circuit = IVCStepCircuit {
            prev_state: initial,
            new_state: output,
            computation_hash: [1u8; 32],
            step_number: 1,
        };
        let pi: Vec<Fr> = circuit.public_inputs();
        let pi_refs: Vec<&[Fr]> = vec![&pi];
        let proof_bytes = prover.pipeline.prove(&circuit, &pi_refs)
            .expect("Proof generation should succeed");

        // Verify the proof
        assert!(prover.pipeline.verify(&proof_bytes, &pi_refs)
            .expect("Verification should complete"));
    }

    #[test]
    fn test_ivc_verify_step_proof() {
        let initial = [0u8; 32];
        let prover = IVCProver::new(initial);

        let mut output = initial;
        output[0] = 1;
        let circuit = IVCStepCircuit {
            prev_state: initial,
            new_state: output,
            computation_hash: [1u8; 32],
            step_number: 1,
        };
        let pi: Vec<helix_circuits::halo2curves::bn256::Fr> = circuit.public_inputs();
        let pi_refs: Vec<&[helix_circuits::halo2curves::bn256::Fr]> = vec![&pi];
        let proof_bytes = prover.pipeline.prove(&circuit, &pi_refs)
            .expect("Proof generation should succeed");

        let step = IVCStep {
            step: 1,
            input_state: initial,
            output_state: output,
            computation_hash: [1u8; 32],
            step_error: 0.001,
            proof: proof_bytes,
        };

        assert!(prover.verify_step_proof(&step));
    }

    #[test]
    fn test_fold_states() {
        let state1 = IVCState {
            step: 5,
            state_commitment: [1; 32],
            accumulated_error: 0.1,
            proof: Some(vec![1, 2, 3]),
            prev_proof_hash: None,
        };
        
        let state2 = IVCState {
            step: 3,
            state_commitment: [2; 32],
            accumulated_error: 0.05,
            proof: Some(vec![4, 5, 6]),
            prev_proof_hash: None,
        };
        
        let combined = fold_states(&state1, &state2);
        
        assert_eq!(combined.step, 8);
        assert!((combined.accumulated_error - 0.15).abs() < 1e-10);
    }
}
