//! On-chain checkpoint attestation during MPC training.
//!
//! Wires together Pedersen commitments, MPC transport, and Ethereum ECDSA signing
//! to produce on-chain checkpoints without reconstructing model weights.
//!
//! # Flow
//!
//! At each checkpoint interval:
//! 1. Each worker computes a Pedersen commitment share over their weight shares
//! 2. Workers exchange commitment shares via the MPC transport
//! 3. Any party combines all shares into the full commitment
//! 4. The combined commitment is hashed to `bytes32` for on-chain storage
//! 5. Each worker signs the attestation message with their Ethereum private key
//! 6. One party collects all signatures and submits on-chain

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::MPCError;
use crate::field::Fr;
use crate::security::commitment::PedersenGenerators;
use crate::session::transport::MPCTransport;
use crate::share_distribution::{
    CheckpointCommitment, CommitmentShare, VectorCommitment, WeightShare,
};
use crate::types::PartyId;

// ============================================================================
// Data structures
// ============================================================================

/// A completed checkpoint ready for on-chain submission.
#[derive(Debug, Clone)]
pub struct OnChainCheckpoint {
    /// Training step at which this checkpoint was taken.
    pub step: usize,
    /// The combined Pedersen commitment bytes32 (for on-chain `weightCommitment`).
    pub commitment_bytes32: [u8; 32],
    /// The combined vector commitment (full detail for off-chain verification).
    pub combined_commitment: VectorCommitment,
    /// Training loss at this checkpoint (publicly revealed).
    pub loss: f64,
    /// Per-party commitment shares (for audit trail).
    pub party_shares: Vec<CommitmentShare>,
}

/// Serializable message for exchanging commitment shares over MPC transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentShareMessage {
    /// The sending party's index.
    pub party_index: usize,
    /// Training step this commitment is for.
    pub step: usize,
    /// Serialized commitment share (via serde).
    pub commitment_share: CommitmentShare,
}

/// Configuration for the checkpoint attestation system.
#[derive(Debug, Clone)]
pub struct CheckpointAttestationConfig {
    /// Pedersen generators (must match across all workers).
    pub generators: PedersenGenerators,
    /// Number of MPC parties.
    pub num_parties: usize,
    /// This party's index.
    pub party_index: usize,
}

// ============================================================================
// Core checkpoint attestation logic
// ============================================================================

/// Manages the checkpoint attestation flow for a single party.
pub struct CheckpointAttestationManager {
    config: CheckpointAttestationConfig,
    checkpoint_system: CheckpointCommitment,
}

impl CheckpointAttestationManager {
    /// Creates a new manager.
    pub fn new(config: CheckpointAttestationConfig) -> Self {
        let checkpoint_system =
            CheckpointCommitment::with_generators(config.generators.clone());
        Self {
            config,
            checkpoint_system,
        }
    }

    /// Computes this party's commitment share for the current weights.
    ///
    /// Returns the commitment share (safe to broadcast — contains only G1 points).
    pub fn compute_local_share(
        &self,
        weight_shares: &[Fr],
        blindings: &[Fr],
    ) -> Result<CommitmentShare, MPCError> {
        let weight_share = WeightShare {
            party: PartyId::from_index(self.config.party_index),
            index: self.config.party_index,
            data: weight_shares.to_vec(),
            shape: vec![weight_shares.len()],
        };
        self.checkpoint_system.compute_share(&weight_share, blindings)
    }

    /// Exchanges commitment shares with all other parties via transport and combines them.
    ///
    /// Each party sends their commitment share to every other party, collects all
    /// shares, and combines them into a single `VectorCommitment`.
    pub async fn exchange_and_combine<T: MPCTransport>(
        &self,
        local_share: &CommitmentShare,
        step: usize,
        transport: &T,
    ) -> Result<(VectorCommitment, Vec<CommitmentShare>), MPCError> {
        let num_parties = self.config.num_parties;
        let my_index = self.config.party_index;

        // Serialize our commitment share.
        let msg = CommitmentShareMessage {
            party_index: my_index,
            step,
            commitment_share: local_share.clone(),
        };
        let msg_bytes = bincode::serialize(&msg)
            .map_err(|e| MPCError::ProtocolError(format!("serialize commitment share: {}", e)))?;

        // Send to all peers and receive from all peers.
        let peers = transport.peers();
        for peer in &peers {
            transport.send(peer, &msg_bytes).await?;
        }

        // Collect all shares (including our own).
        let mut all_shares: Vec<Option<CommitmentShare>> = vec![None; num_parties];
        all_shares[my_index] = Some(local_share.clone());

        for peer in &peers {
            let data = transport.recv(peer).await?;
            let received: CommitmentShareMessage = bincode::deserialize(&data)
                .map_err(|e| {
                    MPCError::ProtocolError(format!(
                        "deserialize commitment share from {:?}: {}",
                        peer, e
                    ))
                })?;

            if received.step != step {
                return Err(MPCError::ProtocolError(format!(
                    "commitment share step mismatch: expected {}, got {}",
                    step, received.step
                )));
            }

            if received.party_index >= num_parties {
                return Err(MPCError::ProtocolError(format!(
                    "invalid party index {} (num_parties={})",
                    received.party_index, num_parties
                )));
            }

            all_shares[received.party_index] = Some(received.commitment_share);
        }

        // Verify all shares collected.
        let collected: Vec<CommitmentShare> = all_shares
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| {
                    MPCError::ProtocolError(format!("missing commitment share from party {}", i))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Combine all commitment shares into the full vector commitment.
        let combined = self.checkpoint_system.combine_shares(&collected)?;

        debug!(
            party = my_index,
            step = step,
            num_shares = collected.len(),
            "Checkpoint commitment shares combined successfully"
        );

        Ok((combined, collected))
    }

    /// Full checkpoint flow: compute local share, exchange, combine.
    ///
    /// Returns an `OnChainCheckpoint` ready for signing and on-chain submission.
    pub async fn create_checkpoint<T: MPCTransport>(
        &self,
        weight_shares: &[Fr],
        blindings: &[Fr],
        step: usize,
        loss: f64,
        transport: &T,
    ) -> Result<OnChainCheckpoint, MPCError> {
        // Step 1: Compute local commitment share.
        let local_share = self.compute_local_share(weight_shares, blindings)?;

        // Step 2: Exchange and combine.
        let (combined, party_shares) =
            self.exchange_and_combine(&local_share, step, transport).await?;

        // Step 3: Hash to bytes32 for on-chain storage.
        let commitment_bytes32 = combined.to_bytes32();

        info!(
            party = self.config.party_index,
            step = step,
            loss = loss,
            "On-chain checkpoint created"
        );

        Ok(OnChainCheckpoint {
            step,
            commitment_bytes32,
            combined_commitment: combined,
            loss,
            party_shares,
        })
    }
}

/// Converts a loss value (f64) to a U256-compatible representation.
///
/// Scales the loss by 1e18 to preserve decimal precision, matching Solidity's
/// convention of using 18-decimal fixed-point values.
pub fn loss_to_u256_scaled(loss: f64) -> u128 {
    let clamped = if loss.is_finite() && loss >= 0.0 {
        loss
    } else if loss.is_finite() {
        0.0
    } else {
        // NaN or Inf — use a sentinel value (1.0 → scales to 1e18).
        1.0
    };
    (clamped * 1e18) as u128
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::transport::LocalTransport;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[tokio::test]
    async fn test_checkpoint_attestation_3_parties() {
        let num_parties = 3;
        let num_weights = 7;
        let step = 10;
        let loss = 0.42;

        let generators = PedersenGenerators::default();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Generate "model weights" (sum of all party shares).
        let weights: Vec<Fr> = (0..num_weights).map(|_| Fr::random(&mut rng)).collect();

        // Generate additive shares.
        let mut shares: Vec<Vec<Fr>> = Vec::new();
        for _ in 0..(num_parties - 1) {
            let share: Vec<Fr> = (0..num_weights).map(|_| Fr::random(&mut rng)).collect();
            shares.push(share);
        }
        let mut last = weights.clone();
        for share in &shares {
            for (j, s) in share.iter().enumerate() {
                last[j] = Fr::sub(&last[j], s);
            }
        }
        shares.push(last);

        // Each party generates independent blinding factors.
        let blindings: Vec<Vec<Fr>> = (0..num_parties)
            .map(|_| (0..num_weights).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        // Create transport mesh.
        let parties: Vec<PartyId> = (0..num_parties).map(PartyId::from_index).collect();
        let transports = LocalTransport::create_mesh(&parties);

        // Spawn each party's checkpoint flow concurrently.
        let mut handles = Vec::new();
        for (i, transport) in transports.into_iter().enumerate() {
            let config = CheckpointAttestationConfig {
                generators: generators.clone(),
                num_parties,
                party_index: i,
            };
            let party_shares = shares[i].clone();
            let party_blindings = blindings[i].clone();

            handles.push(tokio::spawn(async move {
                let manager = CheckpointAttestationManager::new(config);
                manager
                    .create_checkpoint(&party_shares, &party_blindings, step, loss, &transport)
                    .await
            }));
        }

        // Collect results — all parties should produce the same commitment.
        let mut results: Vec<OnChainCheckpoint> = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap().expect("checkpoint should succeed");
            results.push(result);
        }

        // All parties should agree on the commitment bytes32.
        let expected_bytes32 = results[0].commitment_bytes32;
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.commitment_bytes32, expected_bytes32,
                "party {} commitment bytes32 mismatch",
                i
            );
            assert_eq!(r.step, step);
            assert!((r.loss - loss).abs() < 1e-10);
        }

        // Verify the combined commitment is valid (matches the actual weights).
        let total_blindings: Vec<Fr> = (0..num_weights)
            .map(|j| {
                let mut sum = Fr::ZERO;
                for b in &blindings {
                    sum = Fr::add(&sum, &b[j]);
                }
                sum
            })
            .collect();

        let checkpoint_system = CheckpointCommitment::new();
        assert!(
            checkpoint_system.verify_commitment(
                &results[0].combined_commitment,
                &weights,
                &total_blindings
            ),
            "combined commitment should verify against the actual full weights"
        );
    }

    #[test]
    fn test_loss_to_u256_scaled() {
        assert_eq!(loss_to_u256_scaled(1.0), 1_000_000_000_000_000_000);
        assert_eq!(loss_to_u256_scaled(0.5), 500_000_000_000_000_000);
        assert_eq!(loss_to_u256_scaled(0.0), 0);
        assert_eq!(loss_to_u256_scaled(-1.0), 0);
        assert_eq!(loss_to_u256_scaled(f64::NAN), 1_000_000_000_000_000_000);
    }
}
