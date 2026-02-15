//! Aggregator Proof Pipeline — ZK Proof Generation & On-Chain Submission.
//!
//! Connects the aggregator's gradient collection to real ZK proof generation
//! and on-chain proof submission. After the aggregator collects worker gradients
//! and produces an aggregated update, this pipeline:
//!
//! 1. Maintains current model weights and commitment hash
//! 2. Generates a ZK proof of the training step using MLTrainingProverV2
//! 3. Submits the proof to HelixCoordinatorV2 via SCClient
//! 4. Tracks commitment chaining (new_hash → next old_hash)
//! 5. Handles contract responses (accept/reject/slashing)

use std::sync::Arc;
use std::time::{Duration, Instant};

use ethers::types::U256;
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use helix_prover::provers::training_prover_v2::{
    MLTrainingProverV2, TrainingProofResultV2, V2ProverConfig,
};
use helix_circuits::ml::training_step_v2::compute_state_hash_v2;

use crate::sc_client::{SCClient, TrainingProofInputs};

// ============================================================================
// Error Types
// ============================================================================

/// Errors from the aggregator proof pipeline.
#[derive(Error, Debug)]
pub enum ProofPipelineError {
    /// Prover not initialized.
    #[error("Prover not initialized: {0}")]
    NotInitialized(String),

    /// Proof generation failed.
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// On-chain submission failed.
    #[error("On-chain submission failed: {0}")]
    SubmissionFailed(String),

    /// Commitment chaining violation.
    #[error("Commitment mismatch: contract expects {expected}, local has {actual}")]
    CommitmentMismatch { expected: String, actual: String },

    /// Contract rejected the proof (may trigger slashing).
    #[error("Contract rejected proof for round {round_id}: {reason}")]
    ProofRejected { round_id: u64, reason: String },

    /// Model not registered on-chain.
    #[error("Model not registered on-chain")]
    ModelNotRegistered,

    /// Invalid model state.
    #[error("Invalid model state: {0}")]
    InvalidState(String),
}

// ============================================================================
// Model State
// ============================================================================

/// Current model state tracked by the aggregator.
///
/// Maintains weights, commitment hash, and step counter for commitment chaining.
/// After each successful on-chain proof submission, the state is advanced:
/// `new_hash` becomes the next round's `old_hash`.
#[derive(Debug, Clone)]
pub struct AggregatorModelState {
    /// Layer 1 weights: d_hid × d_in (row-major).
    pub w1: Vec<Fr>,
    /// Layer 1 biases: d_hid.
    pub b1: Vec<Fr>,
    /// Layer 2 weights: d_out × d_hid (row-major).
    pub w2: Vec<Fr>,
    /// Layer 2 biases: d_out.
    pub b2: Vec<Fr>,
    /// Current commitment hash (lo, hi) — matches the contract's `currentCommitment`.
    pub commitment: (Fr, Fr),
    /// Current training step number.
    pub step_number: u64,
    /// Accumulated error bound across rounds.
    pub accumulated_error: f64,
}

impl AggregatorModelState {
    /// Creates initial state from model weights.
    pub fn new(
        w1: Vec<Fr>,
        b1: Vec<Fr>,
        w2: Vec<Fr>,
        b2: Vec<Fr>,
    ) -> Self {
        let commitment = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        Self {
            w1,
            b1,
            w2,
            b2,
            commitment,
            step_number: 0,
            accumulated_error: 0.0,
        }
    }

    /// Recomputes commitment from current weights.
    pub fn recompute_commitment(&mut self) {
        self.commitment = compute_state_hash_v2(&self.w1, &self.b1, &self.w2, &self.b2);
    }
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Configuration for the aggregator proof pipeline.
#[derive(Debug, Clone)]
pub struct ProofPipelineConfig {
    /// Model input dimension.
    pub d_in: usize,
    /// Model hidden dimension.
    pub d_hid: usize,
    /// Model output dimension.
    pub d_out: usize,
    /// On-chain model ID.
    pub model_id: u64,
    /// Learning rate for weight updates (as Fr field element).
    pub learning_rate: Fr,
    /// Base error per operation.
    pub base_error: Fr,
    /// Maximum error bound per step (matches contract's `maxErrorBound`).
    pub max_error_bound: Fr,
    /// V2 prover configuration.
    pub prover_config: V2ProverConfig,
    /// Representative training input (d_in elements).
    /// Used as default training data when workers don't provide theirs.
    pub default_input: Vec<Fr>,
    /// Representative training target (d_out elements).
    pub default_target: Vec<Fr>,
    /// Checkpoint proving interval (1 = every step generates a proof).
    pub checkpoint_interval: u64,
}

impl ProofPipelineConfig {
    /// Creates a config for small test models (d_in=2, d_hid=2, d_out=1).
    pub fn test_small() -> Self {
        Self {
            d_in: 2,
            d_hid: 2,
            d_out: 1,
            model_id: 0,
            learning_rate: Fr::from(1u64),
            base_error: Fr::from(1u64),
            max_error_bound: Fr::from(1_000_000_000_000_000_000u64), // 1e18
            prover_config: V2ProverConfig {
                k: 14,
                relu_range: 256,
                exp_range: 256,
                exp_scale: 1000,
                use_freivalds: false,
                self_verify: false,
                use_witness_cache: false,
                ..V2ProverConfig::default()
            },
            default_input: vec![Fr::from(1u64), Fr::from(1u64)],
            default_target: vec![Fr::from(5u64)],
            checkpoint_interval: 1,
        }
    }
}

// ============================================================================
// Submission Result
// ============================================================================

/// Result of a successful on-chain proof submission.
#[derive(Debug, Clone)]
pub struct ProofSubmissionResult {
    /// Transaction hash.
    pub tx_hash: String,
    /// Block number.
    pub block_number: u64,
    /// Gas used.
    pub gas_used: u64,
    /// Proof size in bytes.
    pub proof_size: usize,
    /// Proof generation time.
    pub proof_generation_time: Duration,
    /// Old commitment (before this step).
    pub old_commitment: (Fr, Fr),
    /// New commitment (after this step).
    pub new_commitment: (Fr, Fr),
    /// Step number.
    pub step_number: u64,
    /// Round ID.
    pub round_id: u64,
}

// ============================================================================
// Aggregator Proof Pipeline
// ============================================================================

/// Pipeline that generates ZK proofs and submits them on-chain.
///
/// Lifecycle:
/// 1. `new()` — creates pipeline with prover and chain client
/// 2. `initialize()` — registers model and stakes on-chain
/// 3. `prove_and_submit()` — generates proof and submits for each round
/// 4. After each round, `model_state` is updated with new weights/commitment
pub struct AggregatorProofPipeline {
    /// ZK prover for generating training step proofs.
    prover: MLTrainingProverV2,
    /// Smart contract client for on-chain interaction.
    sc_client: Arc<SCClient>,
    /// Current model state with commitment tracking.
    model_state: Arc<RwLock<AggregatorModelState>>,
    /// Pipeline configuration.
    config: ProofPipelineConfig,
    /// Total proofs submitted on-chain.
    proofs_submitted: Arc<RwLock<u64>>,
    /// Total proofs rejected by contract.
    proofs_rejected: Arc<RwLock<u64>>,
}

impl AggregatorProofPipeline {
    /// Creates a new proof pipeline.
    ///
    /// Initializes `MLTrainingProverV2` with the configured model dimensions
    /// and prover settings. This performs key generation (slow on first call,
    /// cached SRS on subsequent calls).
    pub fn new(
        sc_client: Arc<SCClient>,
        initial_state: AggregatorModelState,
        config: ProofPipelineConfig,
    ) -> Self {
        info!(
            d_in = config.d_in,
            d_hid = config.d_hid,
            d_out = config.d_out,
            k = config.prover_config.k,
            "Initializing aggregator proof pipeline (key generation may take a moment)"
        );

        let prover = MLTrainingProverV2::with_config(
            config.d_in,
            config.d_hid,
            config.d_out,
            config.prover_config.clone(),
        );

        info!("Aggregator proof pipeline initialized");

        Self {
            prover,
            sc_client,
            model_state: Arc::new(RwLock::new(initial_state)),
            config,
            proofs_submitted: Arc::new(RwLock::new(0)),
            proofs_rejected: Arc::new(RwLock::new(0)),
        }
    }

    /// Returns the current model state.
    pub async fn model_state(&self) -> AggregatorModelState {
        self.model_state.read().await.clone()
    }

    /// Returns the current commitment hash.
    pub async fn current_commitment(&self) -> (Fr, Fr) {
        self.model_state.read().await.commitment
    }

    /// Returns the current step number.
    pub async fn current_step(&self) -> u64 {
        self.model_state.read().await.step_number
    }

    /// Returns proof submission statistics.
    pub async fn stats(&self) -> (u64, u64) {
        let submitted = *self.proofs_submitted.read().await;
        let rejected = *self.proofs_rejected.read().await;
        (submitted, rejected)
    }

    /// Generates a ZK proof of the current training step and submits it on-chain.
    ///
    /// This is the main entry point for the proof pipeline. It:
    /// 1. Reads the current model state (weights + commitment)
    /// 2. Builds a circuit witness using `MLTrainingProverV2::build_witness()`
    /// 3. Generates a Halo2 KZG proof (1856-byte EVM-compatible format)
    /// 4. Converts public inputs to U256 for on-chain submission
    /// 5. Submits via `SCClient::submit_proof()`
    /// 6. On success: advances model state (new weights + new commitment)
    /// 7. On failure: logs error and keeps old state
    ///
    /// The `training_input` and `training_target` are the training data for this
    /// step. If None, uses the configured defaults.
    pub async fn prove_and_submit(
        &self,
        round_id: u64,
        training_input: Option<&[Fr]>,
        training_target: Option<&[Fr]>,
    ) -> Result<ProofSubmissionResult, ProofPipelineError> {
        let start = Instant::now();

        // 1. Read current model state
        let state = self.model_state.read().await.clone();
        let step_number = state.step_number + 1;

        let x = training_input.unwrap_or(&self.config.default_input);
        let target = training_target.unwrap_or(&self.config.default_target);

        info!(
            round_id,
            step_number,
            "Generating ZK proof for aggregated training step"
        );

        // 2. Build witness (computes forward + backward + weight update internally)
        let model_id_bytes = {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&self.config.model_id.to_le_bytes());
            bytes
        };

        let witness = MLTrainingProverV2::build_witness_with_params(
            self.config.d_in,
            self.config.d_hid,
            self.config.d_out,
            x,
            target,
            &state.w1,
            &state.b1,
            &state.w2,
            &state.b2,
            self.config.learning_rate,
            step_number,
            self.config.base_error,
            model_id_bytes,
            self.config.max_error_bound,
        );

        // Verify commitment chaining: witness.old_state_hash must match our tracked commitment
        if state.step_number > 0 && witness.old_state_hash != state.commitment {
            return Err(ProofPipelineError::CommitmentMismatch {
                expected: format!("{:?}", state.commitment),
                actual: format!("{:?}", witness.old_state_hash),
            });
        }

        // 3. Generate proof
        let proof_result = self.prover.prove(&witness).map_err(|e| {
            error!(round_id, step_number, error = %e, "Proof generation failed");
            ProofPipelineError::ProofGenerationFailed(e.to_string())
        })?;

        let proof_gen_time = proof_result.generation_time;
        let proof_size = proof_result.proof.len();

        info!(
            round_id,
            step_number,
            proof_size,
            proof_gen_time_ms = proof_gen_time.as_millis() as u64,
            "ZK proof generated successfully"
        );

        // 4. Convert public inputs to U256 for on-chain submission
        let pi = &proof_result.public_inputs;
        if pi.len() < 8 {
            return Err(ProofPipelineError::ProofGenerationFailed(format!(
                "Expected 8 public inputs, got {}",
                pi.len()
            )));
        }

        let inputs = TrainingProofInputs {
            old_hash_lo: fr_to_u256(&pi[0]),
            old_hash_hi: fr_to_u256(&pi[1]),
            new_hash_lo: fr_to_u256(&pi[2]),
            new_hash_hi: fr_to_u256(&pi[3]),
            loss: fr_to_u256(&pi[4]),
            error_bound: fr_to_u256(&pi[5]),
            step_number: fr_to_u256(&pi[6]),
            error_checksum: fr_to_u256(&pi[7]),
        };

        // 5. Submit proof on-chain
        info!(
            round_id,
            model_id = self.config.model_id,
            "Submitting proof on-chain"
        );

        let receipt = self
            .sc_client
            .submit_proof(
                self.config.model_id,
                round_id,
                proof_result.proof.clone(),
                &inputs,
            )
            .await
            .map_err(|e| {
                error!(
                    round_id,
                    model_id = self.config.model_id,
                    error = %e,
                    "On-chain proof submission failed"
                );
                // Check if this is a revert (proof rejected → possible slashing)
                let err_str = e.to_string();
                if err_str.contains("revert") || err_str.contains("InvalidProof") {
                    *self.proofs_rejected.blocking_write() += 1;
                    ProofPipelineError::ProofRejected {
                        round_id,
                        reason: err_str,
                    }
                } else {
                    ProofPipelineError::SubmissionFailed(err_str)
                }
            })?;

        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let block_number = receipt.block_number.map(|b| b.as_u64()).unwrap_or(0);
        let gas_used = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);

        // Check for slashing events in the receipt. The contract does NOT revert
        // on invalid proofs — instead it calls _slash() and returns normally.
        // We detect this by looking for the Slashed event topic:
        // Slashed(address indexed prover, uint256 indexed modelId, uint256 roundId,
        //         uint256 amount, uint256 remainingStake, string reason)
        let slashed_topic = ethers::utils::keccak256(
            b"Slashed(address,uint256,uint256,uint256,uint256,string)",
        );
        let was_slashed = receipt.logs.iter().any(|log| {
            !log.topics.is_empty() && log.topics[0].as_bytes() == slashed_topic
        });

        if was_slashed {
            error!(
                round_id,
                model_id = self.config.model_id,
                tx_hash = %tx_hash,
                "Proof was rejected on-chain (prover slashed)"
            );
            *self.proofs_rejected.write().await += 1;
            return Err(ProofPipelineError::ProofRejected {
                round_id,
                reason: "Proof rejected by verifier — prover was slashed".to_string(),
            });
        }

        info!(
            round_id,
            step_number,
            tx_hash = %tx_hash,
            block_number,
            gas_used,
            total_time_ms = start.elapsed().as_millis() as u64,
            "Proof accepted on-chain"
        );

        // 6. Advance model state with new weights and commitment
        let old_commitment = state.commitment;
        let new_commitment = witness.new_state_hash;

        {
            let mut state = self.model_state.write().await;
            state.w1 = witness.w1_new.clone();
            state.b1 = witness.b1_new.clone();
            state.w2 = witness.w2_new.clone();
            state.b2 = witness.b2_new.clone();
            state.commitment = new_commitment;
            state.step_number = step_number;
            // Track accumulated error (from proof's error_bound field)
            let error_repr = proof_result.total_error.to_repr();
            let error_u64 = u64::from_le_bytes(
                error_repr.as_ref()[..8].try_into().unwrap_or([0; 8]),
            );
            state.accumulated_error += error_u64 as f64;
        }

        *self.proofs_submitted.write().await += 1;

        Ok(ProofSubmissionResult {
            tx_hash,
            block_number,
            gas_used,
            proof_size,
            proof_generation_time: proof_gen_time,
            old_commitment,
            new_commitment,
            step_number,
            round_id,
        })
    }

    /// Generates a ZK proof for a checkpoint and submits it on-chain.
    ///
    /// Unlike `prove_and_submit` which proves a single step from current weights,
    /// this proves the state transition from explicit checkpoint-start weights to
    /// the current weights, with accumulated error covering the full interval.
    pub async fn prove_and_submit_checkpoint(
        &self,
        round_id: u64,
        checkpoint_w1: &[Fr],
        checkpoint_b1: &[Fr],
        checkpoint_w2: &[Fr],
        checkpoint_b2: &[Fr],
        checkpoint_step: u64,
        accumulated_error: Fr,
        training_input: Option<&[Fr]>,
        training_target: Option<&[Fr]>,
    ) -> Result<ProofSubmissionResult, ProofPipelineError> {
        let start = Instant::now();
        let x = training_input.unwrap_or(&self.config.default_input);
        let target = training_target.unwrap_or(&self.config.default_target);

        let model_id_bytes = {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&self.config.model_id.to_le_bytes());
            bytes
        };

        info!(
            round_id,
            checkpoint_step,
            "Generating checkpoint ZK proof"
        );

        // Read current (post-training) weights for the checkpoint end state
        let current_state = self.model_state.read().await.clone();

        // Build witness using checkpoint-start weights
        let witness = MLTrainingProverV2::build_checkpoint_witness(
            self.config.d_in,
            self.config.d_hid,
            self.config.d_out,
            x,
            target,
            checkpoint_w1,
            checkpoint_b1,
            checkpoint_w2,
            checkpoint_b2,
            // new weights (reserved, unused by circuit)
            &current_state.w1,
            &current_state.b1,
            &current_state.w2,
            &current_state.b2,
            self.config.learning_rate,
            checkpoint_step,
            accumulated_error,
            model_id_bytes,
            self.config.max_error_bound,
        );

        // Generate proof
        let proof_result = self.prover.prove(&witness).map_err(|e| {
            error!(round_id, checkpoint_step, error = %e, "Checkpoint proof generation failed");
            ProofPipelineError::ProofGenerationFailed(e.to_string())
        })?;

        let proof_gen_time = proof_result.generation_time;
        let proof_size = proof_result.proof.len();

        info!(
            round_id,
            checkpoint_step,
            proof_size,
            proof_gen_time_ms = proof_gen_time.as_millis() as u64,
            "Checkpoint ZK proof generated"
        );

        // Convert public inputs to U256 for on-chain submission
        let pi = &proof_result.public_inputs;
        if pi.len() < 8 {
            return Err(ProofPipelineError::ProofGenerationFailed(format!(
                "Expected 8 public inputs, got {}",
                pi.len()
            )));
        }

        let inputs = TrainingProofInputs {
            old_hash_lo: fr_to_u256(&pi[0]),
            old_hash_hi: fr_to_u256(&pi[1]),
            new_hash_lo: fr_to_u256(&pi[2]),
            new_hash_hi: fr_to_u256(&pi[3]),
            loss: fr_to_u256(&pi[4]),
            error_bound: fr_to_u256(&pi[5]),
            step_number: fr_to_u256(&pi[6]),
            error_checksum: fr_to_u256(&pi[7]),
        };

        info!(
            round_id,
            model_id = self.config.model_id,
            "Submitting checkpoint proof on-chain"
        );

        let receipt = self
            .sc_client
            .submit_proof(
                self.config.model_id,
                round_id,
                proof_result.proof.clone(),
                &inputs,
            )
            .await
            .map_err(|e| {
                error!(round_id, error = %e, "On-chain checkpoint submission failed");
                ProofPipelineError::SubmissionFailed(e.to_string())
            })?;

        let tx_hash = format!("{:?}", receipt.transaction_hash);
        let block_number = receipt.block_number.map(|b| b.as_u64()).unwrap_or(0);
        let gas_used = receipt.gas_used.map(|g| g.as_u64()).unwrap_or(0);

        let old_commitment =
            compute_state_hash_v2(checkpoint_w1, checkpoint_b1, checkpoint_w2, checkpoint_b2);
        let new_commitment = witness.new_state_hash;

        // Advance model state
        {
            let mut state = self.model_state.write().await;
            state.w1 = witness.w1_new.clone();
            state.b1 = witness.b1_new.clone();
            state.w2 = witness.w2_new.clone();
            state.b2 = witness.b2_new.clone();
            state.commitment = new_commitment;
            state.step_number = checkpoint_step;
        }

        *self.proofs_submitted.write().await += 1;

        info!(
            round_id,
            checkpoint_step,
            tx_hash = %tx_hash,
            block_number,
            gas_used,
            total_time_ms = start.elapsed().as_millis() as u64,
            "Checkpoint proof accepted on-chain"
        );

        Ok(ProofSubmissionResult {
            tx_hash,
            block_number,
            gas_used,
            proof_size,
            proof_generation_time: proof_gen_time,
            old_commitment,
            new_commitment,
            step_number: checkpoint_step,
            round_id,
        })
    }

    /// Queries the on-chain model commitment and verifies it matches local state.
    ///
    /// Returns `true` if commitments match, `false` if there's a divergence.
    /// A mismatch indicates the aggregator's state is out of sync with the contract.
    pub async fn verify_commitment_sync(&self) -> Result<bool, ProofPipelineError> {
        let model_state = self
            .sc_client
            .get_model_state(self.config.model_id)
            .await
            .map_err(|e| ProofPipelineError::SubmissionFailed(e.to_string()))?;

        let local_state = self.model_state.read().await;
        let (lo, hi) = local_state.commitment;

        // Convert local Fr commitment to U256 for comparison
        let local_lo = fr_to_u256(&lo);
        let local_hi = fr_to_u256(&hi);

        // The contract stores commitment as keccak256(abi.encodePacked(lo, hi))
        // We compare against the currentCommitment which is this hash
        let local_hash = compute_hash_pair(local_lo, local_hi);

        // For step 0, the initial commitment is set at registration time
        // For subsequent steps, verify the commitment matches
        if local_state.step_number == 0 {
            return Ok(true);
        }

        let matches = model_state.current_commitment == local_hash;
        if !matches {
            warn!(
                local_commitment = ?local_hash,
                chain_commitment = ?model_state.current_commitment,
                step = local_state.step_number,
                "Commitment mismatch between local state and on-chain"
            );
        }

        Ok(matches)
    }

    /// Registers a new model on-chain and returns the model ID.
    ///
    /// The initial commitment is `keccak256(abi.encodePacked(lo, hi))` matching
    /// the contract's `_hashPair()` function. This ensures the first proof's
    /// `old_hash` commitment will pass the on-chain commitment chaining check.
    pub async fn register_model(&self) -> Result<u64, ProofPipelineError> {
        let state = self.model_state.read().await;
        let (lo, hi) = state.commitment;
        let initial_commitment = compute_hash_pair(fr_to_u256(&lo), fr_to_u256(&hi));

        let min_stake = U256::from(1_000_000_000_000_000_000u64); // 1 ETH

        let model_arch = (
            self.config.d_in as u32,
            self.config.d_hid as u32,
            self.config.d_out as u32,
            2u32, // num_layers (2-layer MLP)
            0u8,  // activation_type (ReLU)
        );

        let (_receipt, model_id) = self
            .sc_client
            .register_model("helix-model", initial_commitment, min_stake, model_arch)
            .await
            .map_err(|e| ProofPipelineError::SubmissionFailed(e.to_string()))?;

        info!(model_id, "Model registered on-chain");
        Ok(model_id)
    }

    /// Stakes ETH for the configured model.
    pub async fn stake(&self, amount: U256) -> Result<(), ProofPipelineError> {
        self.sc_client
            .stake(self.config.model_id, amount)
            .await
            .map_err(|e| ProofPipelineError::SubmissionFailed(e.to_string()))?;

        info!(model_id = self.config.model_id, "Staked for model");
        Ok(())
    }

    /// Starts a new training round on-chain.
    pub async fn start_round(&self, duration_secs: u64) -> Result<(), ProofPipelineError> {
        self.sc_client
            .start_round(self.config.model_id, duration_secs)
            .await
            .map_err(|e| ProofPipelineError::SubmissionFailed(e.to_string()))?;

        info!(
            model_id = self.config.model_id,
            duration_secs, "Training round started on-chain"
        );
        Ok(())
    }

    /// Returns a reference to the SCClient.
    pub fn sc_client(&self) -> &Arc<SCClient> {
        &self.sc_client
    }

    /// Returns a reference to the prover.
    pub fn prover(&self) -> &MLTrainingProverV2 {
        &self.prover
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Converts an Fr field element to ethers U256 (little-endian repr).
fn fr_to_u256(fr: &Fr) -> U256 {
    let repr = fr.to_repr();
    U256::from_little_endian(repr.as_ref())
}

/// Computes `keccak256(abi.encodePacked(lo, hi))` matching the contract's `_hashPair()`.
fn compute_hash_pair(lo: U256, hi: U256) -> U256 {
    let mut data = [0u8; 64];
    lo.to_big_endian(&mut data[0..32]);
    hi.to_big_endian(&mut data[32..64]);
    let hash = ethers::utils::keccak256(&data);
    U256::from_big_endian(&hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_state_commitment() {
        let w1 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)];
        let b1 = vec![Fr::from(0u64), Fr::from(0u64)];
        let w2 = vec![Fr::from(1u64), Fr::from(1u64)];
        let b2 = vec![Fr::from(0u64)];

        let state = AggregatorModelState::new(w1.clone(), b1.clone(), w2.clone(), b2.clone());

        // Commitment should be non-zero
        assert_ne!(state.commitment.0, Fr::ZERO);
        assert_eq!(state.step_number, 0);

        // Recomputing should give the same result
        let expected = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        assert_eq!(state.commitment, expected);
    }

    #[test]
    fn test_fr_to_u256_roundtrip() {
        let fr = Fr::from(42u64);
        let u = fr_to_u256(&fr);
        assert_eq!(u, U256::from(42));

        let fr_zero = Fr::ZERO;
        let u_zero = fr_to_u256(&fr_zero);
        assert_eq!(u_zero, U256::zero());
    }

    #[test]
    fn test_pipeline_config_test_small() {
        let config = ProofPipelineConfig::test_small();
        assert_eq!(config.d_in, 2);
        assert_eq!(config.d_hid, 2);
        assert_eq!(config.d_out, 1);
        assert_eq!(config.default_input.len(), 2);
        assert_eq!(config.default_target.len(), 1);
    }
}
