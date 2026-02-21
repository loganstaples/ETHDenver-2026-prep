//! On-chain checkpoint submission orchestrator.
//!
//! Bridges MPC training checkpoints to the HelixCoordinatorV4 contract.
//! Given checkpoint data from the MPC training loop (commitment bytes32 + loss),
//! this module handles:
//!
//! 1. Signing the checkpoint attestation message with each worker's Ethereum wallet
//! 2. Collecting all ECDSA signatures
//! 3. Submitting the multi-party attestation to the V4 contract
//!
//! # Usage
//!
//! ```ignore
//! let receipts = submit_all_checkpoints(
//!     &chain_client,
//!     job_id,
//!     &checkpoints,
//!     &worker_wallets,
//! ).await?;
//! ```

use anyhow::{anyhow, Result};
use ethers::signers::{LocalWallet, Signer};
use ethers::types::{Bytes, TransactionReceipt, U256};
use tracing::info;

use crate::rpc::chain_v4::{sign_checkpoint, ChainClientV4};

/// Result of a single on-chain checkpoint submission.
#[derive(Debug)]
pub struct CheckpointSubmission {
    /// Training step number.
    pub step: u64,
    /// Transaction receipt from the on-chain submission.
    pub receipt: TransactionReceipt,
    /// Number of signatures included in the submission.
    pub signer_count: usize,
}

/// Checkpoint data required for on-chain submission.
///
/// This is a minimal struct that decouples the submission logic from
/// MPC-internal types. Callers can construct this from any source
/// (training results, manual data, etc.).
#[derive(Debug, Clone)]
pub struct CheckpointData {
    /// Training step at which this checkpoint was taken.
    pub step: u64,
    /// Combined Pedersen commitment hash (bytes32).
    pub commitment_bytes32: [u8; 32],
    /// Loss value (will be scaled to 1e18 for on-chain representation).
    pub loss: f64,
    /// Optional ZK proof bytes (serialized Halo2 SHPLONK proof).
    pub proof: Option<Vec<u8>>,
    /// Optional public inputs for the ZK proof (6 elements for StateTransitionCircuit).
    pub public_inputs: Option<Vec<U256>>,
}

/// Sign a checkpoint attestation message with all worker wallets and submit to the V4 contract.
///
/// Each wallet produces an ECDSA signature over:
/// `keccak256(abi.encodePacked("HELIX_CHECKPOINT", jobId, stepNumber, weightCommitment, loss))`
///
/// The V4 contract verifies all signatures belong to active workers before accepting.
pub async fn sign_and_submit_checkpoint(
    chain_client: &ChainClientV4,
    job_id: u64,
    checkpoint: &CheckpointData,
    worker_wallets: &[LocalWallet],
) -> Result<CheckpointSubmission> {
    let loss_u256 = loss_to_u256(checkpoint.loss);

    let (receipt, signer_count) = if let (Some(proof), Some(pub_inputs)) =
        (&checkpoint.proof, &checkpoint.public_inputs)
    {
        // Submit with ZK proof — the proof is the attestation, no signatures needed.
        info!(
            job_id = job_id,
            step = checkpoint.step,
            proof_size = proof.len(),
            pub_inputs = pub_inputs.len(),
            "Submitting checkpoint on-chain with ZK proof"
        );

        let r = chain_client
            .submit_checkpoint_with_proof(
                job_id,
                checkpoint.step,
                checkpoint.commitment_bytes32,
                loss_u256,
                proof.clone(),
                pub_inputs.clone(),
            )
            .await
            .map_err(|e| {
                anyhow!(
                    "submit_checkpoint_with_proof on-chain failed (job={}, step={}): {}",
                    job_id,
                    checkpoint.step,
                    e
                )
            })?;
        (r, 0usize) // No signers when using ZK proof
    } else {
        // Submit with multi-party signatures.
        let job_id_u256 = U256::from(job_id);
        let step_u256 = U256::from(checkpoint.step);

        let mut signatures: Vec<Bytes> = Vec::with_capacity(worker_wallets.len());
        for wallet in worker_wallets {
            let sig = sign_checkpoint(
                wallet,
                job_id_u256,
                step_u256,
                checkpoint.commitment_bytes32,
                loss_u256,
            )
            .await
            .map_err(|e| anyhow!("worker {} sign failed: {}", wallet.address(), e))?;
            signatures.push(sig);
        }

        let count = signatures.len();

        info!(
            job_id = job_id,
            step = checkpoint.step,
            signers = count,
            "Submitting checkpoint on-chain with signatures"
        );

        let r = chain_client
            .submit_checkpoint(
                job_id,
                checkpoint.step,
                checkpoint.commitment_bytes32,
                loss_u256,
                signatures,
            )
            .await
            .map_err(|e| {
                anyhow!(
                    "submit_checkpoint on-chain failed (job={}, step={}): {}",
                    job_id,
                    checkpoint.step,
                    e
                )
            })?;
        (r, count)
    };

    info!(
        job_id = job_id,
        step = checkpoint.step,
        tx_hash = ?receipt.transaction_hash,
        gas_used = ?receipt.gas_used,
        "Checkpoint submitted on-chain"
    );

    Ok(CheckpointSubmission {
        step: checkpoint.step,
        receipt,
        signer_count,
    })
}

/// Submit all checkpoints from a training run to the V4 contract.
///
/// Processes checkpoints sequentially (on-chain nonce ordering requires this).
/// Returns a `CheckpointSubmission` for each successfully submitted checkpoint.
pub async fn submit_all_checkpoints(
    chain_client: &ChainClientV4,
    job_id: u64,
    checkpoints: &[CheckpointData],
    worker_wallets: &[LocalWallet],
) -> Result<Vec<CheckpointSubmission>> {
    if checkpoints.is_empty() {
        return Ok(Vec::new());
    }

    info!(
        job_id = job_id,
        num_checkpoints = checkpoints.len(),
        num_signers = worker_wallets.len(),
        "Submitting training checkpoints on-chain"
    );

    let mut submissions = Vec::with_capacity(checkpoints.len());
    for checkpoint in checkpoints {
        let submission =
            sign_and_submit_checkpoint(chain_client, job_id, checkpoint, worker_wallets).await?;
        submissions.push(submission);
    }

    info!(
        job_id = job_id,
        submitted = submissions.len(),
        "All checkpoints submitted on-chain"
    );

    Ok(submissions)
}

/// Convert a loss value (f64) to a U256 with 1e18 scaling.
///
/// Matches the convention used in the V4 contract for fixed-point loss values.
fn loss_to_u256(loss: f64) -> U256 {
    let clamped = if loss.is_finite() && loss >= 0.0 {
        loss
    } else if loss.is_finite() {
        0.0
    } else {
        1.0 // NaN or Inf sentinel
    };
    U256::from((clamped * 1e18) as u128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_to_u256() {
        assert_eq!(loss_to_u256(1.0), U256::from(1_000_000_000_000_000_000u128));
        assert_eq!(loss_to_u256(0.5), U256::from(500_000_000_000_000_000u128));
        assert_eq!(loss_to_u256(0.0), U256::zero());
        assert_eq!(loss_to_u256(-1.0), U256::zero());
        assert_eq!(
            loss_to_u256(f64::NAN),
            U256::from(1_000_000_000_000_000_000u128)
        );
    }

    #[test]
    fn test_checkpoint_data_construction() {
        let data = CheckpointData {
            step: 25,
            commitment_bytes32: [0xAB; 32],
            loss: 0.42,
            proof: None,
            public_inputs: None,
        };
        assert_eq!(data.step, 25);
        assert_eq!(data.commitment_bytes32, [0xAB; 32]);
        assert!((data.loss - 0.42).abs() < 1e-10);
    }
}
