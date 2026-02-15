//! Encrypted share distribution and reconstruction over TCP.
//!
//! This module implements the complete network protocol for the MPC weight
//! lifecycle as described in the HELIX system design:
//!
//! 1. **Distribution**: Owner connects to each worker over TCP, sends encrypted
//!    weight shares + blinding shares + initial Pedersen commitment C0.
//! 2. **Commitment Verification**: Workers decrypt shares, compute commitment
//!    shares using the owner's blinding shares, send them back. Owner combines
//!    and verifies the combined commitment equals C0.
//! 3. **Reconstruction**: At training completion, owner requests final shares.
//!    Workers encrypt their (potentially modified) shares to the owner, along
//!    with checkpoint commitment data. Owner reconstructs and verifies.
//!
//! # Protocol
//!
//! The owner acts as TCP client, connecting to each worker's listener. Messages
//! use length-prefixed framing (`[4-byte BE length][bincode payload]`), matching
//! the convention used throughout helix-mpc's transport layer.
//!
//! # Security
//!
//! - Weight shares are encrypted with x25519 DH + AES-256-GCM (same scheme as
//!   `share_distribution`). No plaintext share data crosses the wire.
//! - Blinding shares are similarly encrypted so only the intended worker can
//!   decrypt them.
//! - The Pedersen commitment C0 is sent in the clear (it's a binding commitment
//!   that reveals nothing about the weights).
//! - Workers verify their combined commitment shares equal C0, proving the owner
//!   distributed valid additive shares.

use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use x25519_dalek::StaticSecret;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::security::commitment::PedersenGenerators;
use crate::share_distribution::{
    CheckpointCommitment, CommitmentShare, DistributionResult, EncryptedShare,
    ShareDistributor, ShareReceiver, VectorCommitment, WeightReconstructor,
    WeightShare, X25519PublicKey, X25519StaticSecret,
    derive_share_key, derive_share_nonce, encrypt_with_key, decrypt_with_key,
    serialize_fr_vec, deserialize_fr_vec, encrypt_share_for_owner,
    generate_x25519_keypair,
};
use crate::types::PartyId;

/// Maximum allowed message size (64 MB) to prevent OOM from malicious length prefixes.
const MAX_MSG_SIZE: usize = 64 * 1024 * 1024;

// ============================================================================
// Protocol messages
// ============================================================================

/// Encrypted blinding share payload (same DH+AES-GCM scheme as weight shares).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedBlindings {
    /// Sender's ephemeral x25519 public key (32 bytes).
    pub ephemeral_public_key: [u8; 32],
    /// AES-256-GCM ciphertext of serialized Fr blinding elements.
    pub ciphertext: Vec<u8>,
    /// Number of Fr elements (one per weight element).
    pub num_elements: usize,
}

/// Serialized Fr vector for transferring blinding factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrVecPayload {
    /// Raw bytes: 32 bytes per Fr element, little-endian.
    pub data: Vec<u8>,
    /// Number of Fr elements.
    pub count: usize,
}

impl FrVecPayload {
    /// Creates a payload from a Vec<Fr>.
    pub fn from_fr_vec(values: &[Fr]) -> Self {
        Self {
            data: serialize_fr_vec(values),
            count: values.len(),
        }
    }

    /// Converts back to Vec<Fr>.
    pub fn to_fr_vec(&self) -> MPCResult<Vec<Fr>> {
        deserialize_fr_vec(&self.data, self.count)
    }
}

/// Messages exchanged between owner and workers during the distribution and
/// reconstruction protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtocolMessage {
    /// Owner → Worker: initial share distribution.
    ///
    /// Contains the encrypted weight share, encrypted blinding share (so the
    /// worker can compute a commitment share that combines to C0), the initial
    /// commitment C0, and the Pedersen generators (must match across all parties).
    ShareDistribution {
        /// The worker's encrypted weight share.
        encrypted_share: EncryptedShare,
        /// The worker's encrypted blinding share (additive share of owner's blindings).
        encrypted_blindings: EncryptedBlindings,
        /// The owner's initial Pedersen commitment to the full weights.
        initial_commitment: VectorCommitment,
        /// Pedersen generators (must be identical for all participants).
        generators: PedersenGenerators,
    },

    /// Worker → Owner: commitment share for verification.
    ///
    /// After decrypting their weight share and blinding share, the worker
    /// computes their Pedersen commitment share and sends it back.
    CommitmentShareResponse {
        /// The worker's Pedersen commitment share.
        commitment_share: CommitmentShare,
    },

    /// Owner → Worker: result of commitment verification.
    ///
    /// The owner combines all commitment shares and checks whether the combined
    /// commitment equals C0. If verified, training can proceed.
    CommitmentVerified {
        /// Whether the combined commitment matched C0.
        verified: bool,
    },

    /// Owner → Worker: request for final weight shares.
    ///
    /// Sent at training completion. Includes the owner's x25519 public key so
    /// workers can encrypt their final shares to it.
    ReconstructionRequest {
        /// The owner's x25519 public key for encrypting final shares.
        owner_public_key: [u8; 32],
    },

    /// Worker → Owner: final encrypted share + checkpoint data.
    ///
    /// Workers encrypt their (potentially trained) weight shares to the owner
    /// and include their Pedersen commitment share + blinding factors for
    /// checkpoint verification.
    FinalShareResponse {
        /// The worker's final weight share, encrypted to the owner.
        encrypted_share: EncryptedShare,
        /// Commitment share for checkpoint verification.
        commitment_share: CommitmentShare,
        /// The worker's blinding factors (for the owner to sum and verify).
        blindings: FrVecPayload,
    },
}

// ============================================================================
// TCP framing helpers
// ============================================================================

/// Sends a protocol message over a TCP stream with length-prefixed framing.
pub async fn send_message(stream: &mut TcpStream, msg: &ProtocolMessage) -> MPCResult<()> {
    let data = bincode::serialize(msg).map_err(|e| {
        MPCError::CommunicationError(format!("serialize failed: {}", e))
    })?;
    if data.len() > MAX_MSG_SIZE {
        return Err(MPCError::CommunicationError(format!(
            "message too large: {} bytes (max {})",
            data.len(),
            MAX_MSG_SIZE
        )));
    }
    let len = (data.len() as u32).to_be_bytes();
    stream.write_all(&len).await.map_err(|e| {
        MPCError::CommunicationError(format!("write len failed: {}", e))
    })?;
    stream.write_all(&data).await.map_err(|e| {
        MPCError::CommunicationError(format!("write body failed: {}", e))
    })?;
    stream.flush().await.map_err(|e| {
        MPCError::CommunicationError(format!("flush failed: {}", e))
    })?;
    Ok(())
}

/// Receives a protocol message from a TCP stream with length-prefixed framing.
pub async fn recv_message(stream: &mut TcpStream) -> MPCResult<ProtocolMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.map_err(|e| {
        MPCError::CommunicationError(format!("read len failed: {}", e))
    })?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(MPCError::CommunicationError(format!(
            "message too large: {} bytes (max {})",
            len, MAX_MSG_SIZE
        )));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await.map_err(|e| {
        MPCError::CommunicationError(format!("read body failed: {}", e))
    })?;
    bincode::deserialize(&buf).map_err(|e| {
        MPCError::CommunicationError(format!("deserialize failed: {}", e))
    })
}

// ============================================================================
// Blinding encryption helpers
// ============================================================================

/// Encrypts a vector of Fr blinding factors to a worker's x25519 public key.
///
/// Uses the same DH+AES-GCM scheme as weight share encryption.
fn encrypt_blindings(
    blindings: &[Fr],
    recipient_pk: &X25519PublicKey,
    rng: &mut impl RngCore,
) -> MPCResult<EncryptedBlindings> {
    let mut ephemeral_bytes = [0u8; 32];
    rng.fill_bytes(&mut ephemeral_bytes);
    let ephemeral_secret = StaticSecret::from(ephemeral_bytes);
    let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

    let shared_secret = ephemeral_secret.diffie_hellman(recipient_pk);
    let aes_key = derive_share_key(shared_secret.as_bytes(), ephemeral_public.as_bytes());
    let nonce = derive_share_nonce(shared_secret.as_bytes(), ephemeral_public.as_bytes());

    let plaintext = serialize_fr_vec(blindings);
    let ciphertext = encrypt_with_key(&aes_key, &plaintext, &nonce)?;

    Ok(EncryptedBlindings {
        ephemeral_public_key: *ephemeral_public.as_bytes(),
        ciphertext,
        num_elements: blindings.len(),
    })
}

/// Decrypts blinding factors using the worker's x25519 private key.
fn decrypt_blindings(
    encrypted: &EncryptedBlindings,
    secret_key: &StaticSecret,
) -> MPCResult<Vec<Fr>> {
    let ephemeral_pk = X25519PublicKey::from(encrypted.ephemeral_public_key);
    let shared_secret = secret_key.diffie_hellman(&ephemeral_pk);

    let aes_key = derive_share_key(
        shared_secret.as_bytes(),
        &encrypted.ephemeral_public_key,
    );
    let nonce = derive_share_nonce(
        shared_secret.as_bytes(),
        &encrypted.ephemeral_public_key,
    );

    let plaintext = decrypt_with_key(&aes_key, &encrypted.ciphertext, &nonce)?;
    deserialize_fr_vec(&plaintext, encrypted.num_elements)
}

// ============================================================================
// Owner-side session
// ============================================================================

/// Result of distributing shares over the network.
pub struct NetworkDistributionResult {
    /// The underlying in-memory distribution result (encrypted shares, C0, blindings).
    pub distribution: DistributionResult,
    /// The combined commitment from all workers (should equal C0).
    pub combined_commitment: VectorCommitment,
    /// Whether the commitment verification passed.
    pub verified: bool,
    /// Per-worker TCP streams (kept open for reconstruction).
    pub worker_streams: Vec<(PartyId, TcpStream)>,
}

/// Distributes model weights to workers over TCP.
///
/// This is the owner-side entry point for the distribution phase:
/// 1. Splits weights into additive shares and encrypts each to a worker
/// 2. Splits owner's Pedersen blindings into additive shares
/// 3. Connects to each worker via TCP and sends encrypted data + C0
/// 4. Collects commitment shares from workers and verifies combined == C0
///
/// Returns the distribution result, combined commitment, verification status,
/// and open TCP streams for later reconstruction.
pub async fn distribute_shares(
    weights: &[f64],
    shape: &[usize],
    workers: &[(PartyId, SocketAddr, X25519PublicKey)],
    seed: Option<u64>,
) -> MPCResult<NetworkDistributionResult> {
    let n = workers.len();
    if n < 2 {
        return Err(MPCError::InsufficientParties {
            required: 2,
            available: n,
        });
    }

    let worker_pks: Vec<X25519PublicKey> = workers.iter().map(|(_, _, pk)| *pk).collect();
    let worker_ids: Vec<PartyId> = workers.iter().map(|(id, _, _)| id.clone()).collect();

    // Create distributor and perform in-memory distribution
    let mut distributor = match seed {
        Some(s) => ShareDistributor::with_seed(s),
        None => ShareDistributor::new(),
    };
    let result = distributor.distribute(weights, shape, &worker_pks, &worker_ids)?;

    // Split the owner's blindings into additive shares (one per worker)
    let blinding_shares = distributor.generate_additive_shares(&result.blindings, n);

    // Encrypt blinding shares to each worker's public key
    let mut rng = match seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s.wrapping_add(42)),
        None => ChaCha20Rng::from_entropy(),
    };
    let mut encrypted_blinding_shares = Vec::with_capacity(n);
    for (i, blinding_share) in blinding_shares.iter().enumerate() {
        let enc = encrypt_blindings(blinding_share, &worker_pks[i], &mut rng)?;
        encrypted_blinding_shares.push(enc);
    }

    let generators = distributor.generators().clone();

    // Connect to each worker and send distribution messages
    let mut worker_streams = Vec::with_capacity(n);
    let mut commitment_shares = Vec::with_capacity(n);

    for (i, (party_id, addr, _)) in workers.iter().enumerate() {
        let mut stream = TcpStream::connect(addr).await.map_err(|e| {
            MPCError::CommunicationError(format!(
                "failed to connect to worker {} at {}: {}",
                party_id, addr, e
            ))
        })?;

        // Send distribution message
        let msg = ProtocolMessage::ShareDistribution {
            encrypted_share: result.encrypted_shares[i].clone(),
            encrypted_blindings: encrypted_blinding_shares[i].clone(),
            initial_commitment: result.initial_commitment.clone(),
            generators: generators.clone(),
        };
        send_message(&mut stream, &msg).await?;

        // Receive commitment share response
        let response = recv_message(&mut stream).await?;
        match response {
            ProtocolMessage::CommitmentShareResponse { commitment_share } => {
                commitment_shares.push(commitment_share);
            }
            other => {
                return Err(MPCError::ProtocolError(format!(
                    "expected CommitmentShareResponse, got {:?}",
                    std::mem::discriminant(&other)
                )));
            }
        }

        worker_streams.push((party_id.clone(), stream));
    }

    // Combine commitment shares and verify against C0
    let checkpoint = CheckpointCommitment::with_generators(generators);
    let combined = checkpoint.combine_shares(&commitment_shares)?;

    // Verify: combined commitment should equal the initial commitment C0.
    // We compare the aggregate points (EC point equality).
    let verified = combined.aggregate == result.initial_commitment.aggregate
        && combined.element_commitments.len() == result.initial_commitment.element_commitments.len()
        && combined
            .element_commitments
            .iter()
            .zip(result.initial_commitment.element_commitments.iter())
            .all(|(a, b)| a == b);

    // Send verification result to all workers
    for (_, stream) in &mut worker_streams {
        let msg = ProtocolMessage::CommitmentVerified { verified };
        send_message(stream, &msg).await?;
    }

    if !verified {
        return Err(MPCError::CommitmentVerificationFailed {
            party: PartyId::new("combined"),
        });
    }

    Ok(NetworkDistributionResult {
        distribution: result,
        combined_commitment: combined,
        verified,
        worker_streams,
    })
}

/// Reconstructs trained weights from workers over TCP.
///
/// This is the owner-side entry point for the reconstruction phase:
/// 1. Sends reconstruction request with owner's public key to each worker
/// 2. Collects encrypted final shares + commitment data from each worker
/// 3. Combines commitment shares into a checkpoint commitment
/// 4. Decrypts all shares, sums to reconstruct, verifies against checkpoint
///
/// The `streams` should be the open TCP connections from distribution (or new
/// connections if the distribution streams were closed).
pub async fn reconstruct_shares(
    owner_secret: &X25519StaticSecret,
    streams: &mut [(PartyId, TcpStream)],
) -> MPCResult<ReconstructionResult> {
    let owner_public = X25519PublicKey::from(owner_secret);

    let mut encrypted_shares = Vec::with_capacity(streams.len());
    let mut commitment_shares = Vec::with_capacity(streams.len());
    let mut all_blindings: Vec<Vec<Fr>> = Vec::with_capacity(streams.len());

    for (party_id, stream) in streams.iter_mut() {
        // Send reconstruction request
        let msg = ProtocolMessage::ReconstructionRequest {
            owner_public_key: *owner_public.as_bytes(),
        };
        send_message(stream, &msg).await?;

        // Receive final share response
        let response = recv_message(stream).await?;
        match response {
            ProtocolMessage::FinalShareResponse {
                encrypted_share,
                commitment_share,
                blindings,
            } => {
                encrypted_shares.push(encrypted_share);
                commitment_shares.push(commitment_share);
                all_blindings.push(blindings.to_fr_vec()?);
            }
            other => {
                return Err(MPCError::ProtocolError(format!(
                    "expected FinalShareResponse from {}, got {:?}",
                    party_id,
                    std::mem::discriminant(&other)
                )));
            }
        }
    }

    // Combine commitment shares into checkpoint commitment
    let checkpoint = CheckpointCommitment::new();
    let combined_checkpoint = checkpoint.combine_shares(&commitment_shares)?;

    // Compute total blindings (sum per element across all workers)
    let num_elements = all_blindings[0].len();
    let total_blindings: Vec<Fr> = (0..num_elements)
        .map(|j| {
            let mut sum = Fr::ZERO;
            for worker_blindings in &all_blindings {
                sum = Fr::add(&sum, &worker_blindings[j]);
            }
            sum
        })
        .collect();

    // Decrypt all shares, sum, verify against checkpoint
    let reconstructor = WeightReconstructor::new(owner_secret.clone());
    let weights = reconstructor.reconstruct(
        &encrypted_shares,
        Some(&combined_checkpoint),
        Some(&total_blindings),
    )?;

    Ok(ReconstructionResult {
        weights,
        checkpoint_commitment: combined_checkpoint,
        total_blindings,
    })
}

/// Result of reconstructing weights from workers over the network.
pub struct ReconstructionResult {
    /// The reconstructed model weights (f64).
    pub weights: Vec<f64>,
    /// The checkpoint commitment from the final weight shares.
    pub checkpoint_commitment: VectorCommitment,
    /// The combined blinding factors (sum across all workers per element).
    pub total_blindings: Vec<Fr>,
}

// ============================================================================
// Worker-side session
// ============================================================================

/// State held by a worker after receiving their share.
pub struct WorkerShareState {
    /// The decrypted weight share.
    pub weight_share: WeightShare,
    /// The blinding share (for commitment verification).
    pub blinding_share: Vec<Fr>,
    /// The Pedersen generators (shared with all participants).
    pub generators: PedersenGenerators,
    /// The initial commitment C0 from the owner.
    pub initial_commitment: VectorCommitment,
    /// The worker's x25519 secret key (for encrypting shares back to owner).
    pub secret_key: StaticSecret,
}

/// Worker-side handler for the distribution phase.
///
/// Listens on a TCP port for the owner's connection. When connected:
/// 1. Receives the encrypted share + encrypted blindings + C0
/// 2. Decrypts both using the worker's x25519 private key
/// 3. Computes the Pedersen commitment share using the decrypted blinding share
/// 4. Sends the commitment share back to the owner
/// 5. Receives the verification result
///
/// Returns the worker's state (share + blindings + generators) for use during
/// training and later reconstruction.
pub async fn worker_receive_distribution(
    listener: &TcpListener,
    receiver: &ShareReceiver,
    secret_key: &StaticSecret,
) -> MPCResult<(WorkerShareState, TcpStream)> {
    let (mut stream, _addr) = listener.accept().await.map_err(|e| {
        MPCError::CommunicationError(format!("accept failed: {}", e))
    })?;

    // Receive distribution message
    let msg = recv_message(&mut stream).await?;
    let (encrypted_share, encrypted_blindings, initial_commitment, generators) = match msg {
        ProtocolMessage::ShareDistribution {
            encrypted_share,
            encrypted_blindings,
            initial_commitment,
            generators,
        } => (encrypted_share, encrypted_blindings, initial_commitment, generators),
        other => {
            return Err(MPCError::ProtocolError(format!(
                "expected ShareDistribution, got {:?}",
                std::mem::discriminant(&other)
            )));
        }
    };

    // Decrypt weight share
    let weight_share = receiver.receive(&encrypted_share)?;

    // Decrypt blinding share
    let blinding_share = decrypt_blindings(&encrypted_blindings, secret_key)?;

    // Compute commitment share using the owner's blinding share
    let checkpoint = CheckpointCommitment::with_generators(generators.clone());
    let commitment_share = checkpoint.compute_share(&weight_share, &blinding_share)?;

    // Send commitment share back
    let response = ProtocolMessage::CommitmentShareResponse { commitment_share };
    send_message(&mut stream, &response).await?;

    // Receive verification result
    let verify_msg = recv_message(&mut stream).await?;
    match verify_msg {
        ProtocolMessage::CommitmentVerified { verified } => {
            if !verified {
                return Err(MPCError::CommitmentVerificationFailed {
                    party: receiver.party_id().clone(),
                });
            }
        }
        other => {
            return Err(MPCError::ProtocolError(format!(
                "expected CommitmentVerified, got {:?}",
                std::mem::discriminant(&other)
            )));
        }
    }

    let state = WorkerShareState {
        weight_share,
        blinding_share,
        generators,
        initial_commitment,
        secret_key: secret_key.clone(),
    };

    Ok((state, stream))
}

/// Worker-side handler for the reconstruction phase.
///
/// Receives a reconstruction request from the owner, encrypts the worker's
/// final weight share to the owner's public key, computes a fresh Pedersen
/// commitment share with random blindings, and sends everything back.
///
/// The `share` parameter should be the worker's current weight share
/// (potentially modified by training).
pub async fn worker_send_final_share(
    stream: &mut TcpStream,
    share: &WeightShare,
    generators: &PedersenGenerators,
    seed: Option<u64>,
) -> MPCResult<()> {
    // Receive reconstruction request
    let msg = recv_message(stream).await?;
    let owner_pk_bytes = match msg {
        ProtocolMessage::ReconstructionRequest { owner_public_key } => owner_public_key,
        other => {
            return Err(MPCError::ProtocolError(format!(
                "expected ReconstructionRequest, got {:?}",
                std::mem::discriminant(&other)
            )));
        }
    };

    let owner_pk = X25519PublicKey::from(owner_pk_bytes);

    // Encrypt final share to owner
    let mut rng = match seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s),
        None => ChaCha20Rng::from_entropy(),
    };
    let encrypted_share = encrypt_share_for_owner(share, &owner_pk, &mut rng)?;

    // Compute checkpoint commitment share with fresh random blindings
    let blindings: Vec<Fr> = (0..share.data.len())
        .map(|_| Fr::random(&mut rng))
        .collect();

    let checkpoint = CheckpointCommitment::with_generators(generators.clone());
    let commitment_share = checkpoint.compute_share(share, &blindings)?;

    // Send final share response
    let response = ProtocolMessage::FinalShareResponse {
        encrypted_share,
        commitment_share,
        blindings: FrVecPayload::from_fr_vec(&blindings),
    };
    send_message(stream, &response).await?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_blinding_encryption_roundtrip() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (secret, public) = generate_x25519_keypair(&mut rng);

        let blindings: Vec<Fr> = vec![
            Fr::from_f64(0.1),
            Fr::from_f64(-0.5),
            Fr::from_f64(1.0),
            Fr::ZERO,
            Fr::from_f64(-999.0),
        ];

        let encrypted = encrypt_blindings(&blindings, &public, &mut rng).unwrap();
        assert_eq!(encrypted.num_elements, 5);

        let decrypted = decrypt_blindings(&encrypted, &secret).unwrap();
        assert_eq!(decrypted.len(), blindings.len());

        for (orig, dec) in blindings.iter().zip(decrypted.iter()) {
            assert!(
                orig.ct_eq(dec).to_bool(),
                "blinding roundtrip mismatch"
            );
        }
    }

    #[tokio::test]
    async fn test_fr_vec_payload_roundtrip() {
        let values: Vec<Fr> = vec![
            Fr::from_f64(3.14),
            Fr::ONE,
            Fr::ZERO,
            Fr::from_f64(-2.718),
        ];

        let payload = FrVecPayload::from_fr_vec(&values);
        assert_eq!(payload.count, 4);
        assert_eq!(payload.data.len(), 4 * 32);

        let recovered = payload.to_fr_vec().unwrap();
        assert_eq!(recovered.len(), values.len());
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!(orig.ct_eq(rec).to_bool());
        }
    }

    #[tokio::test]
    async fn test_protocol_message_serialization() {
        // Verify that all protocol message variants can be serialized/deserialized
        let msg = ProtocolMessage::CommitmentVerified { verified: true };
        let bytes = bincode::serialize(&msg).unwrap();
        let recovered: ProtocolMessage = bincode::deserialize(&bytes).unwrap();
        match recovered {
            ProtocolMessage::CommitmentVerified { verified } => assert!(verified),
            _ => panic!("wrong variant"),
        }

        let msg2 = ProtocolMessage::ReconstructionRequest {
            owner_public_key: [0xAB; 32],
        };
        let bytes2 = bincode::serialize(&msg2).unwrap();
        let recovered2: ProtocolMessage = bincode::deserialize(&bytes2).unwrap();
        match recovered2 {
            ProtocolMessage::ReconstructionRequest { owner_public_key } => {
                assert_eq!(owner_public_key, [0xAB; 32]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn test_two_worker_network_distribution_and_reconstruction() {
        // End-to-end test with 2 workers on TCP (localhost).
        let weights = vec![0.1, -0.5, 1.2, 0.0, -3.14, 2.718];
        let shape = vec![2, 3];

        // Setup workers: generate keys and bind listeners
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut worker_info = Vec::new();
        let mut worker_listeners = Vec::new();

        for i in 0..2 {
            let (secret, public) = generate_x25519_keypair(&mut rng);
            let party_id = PartyId::from_index(i);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            worker_info.push((party_id, addr, public, secret));
            worker_listeners.push(listener);
        }

        let owner_workers: Vec<(PartyId, SocketAddr, X25519PublicKey)> = worker_info
            .iter()
            .map(|(id, addr, pk, _)| (id.clone(), *addr, *pk))
            .collect();

        // Spawn worker tasks
        let mut worker_handles = Vec::new();
        for (i, listener) in worker_listeners.into_iter().enumerate() {
            let party_id = worker_info[i].0.clone();
            let secret = worker_info[i].3.clone();
            let receiver = ShareReceiver::new(secret.clone(), party_id);

            let handle = tokio::spawn(async move {
                let (state, mut stream) =
                    worker_receive_distribution(&listener, &receiver, &secret).await.unwrap();

                // Reconstruction: send final share back
                worker_send_final_share(
                    &mut stream,
                    &state.weight_share,
                    &state.generators,
                    Some(500 + i as u64),
                )
                .await
                .unwrap();

                state.weight_share.data.clone()
            });
            worker_handles.push(handle);
        }

        // Owner distributes
        let mut dist_result =
            distribute_shares(&weights, &shape, &owner_workers, Some(100)).await.unwrap();
        assert!(dist_result.verified);

        // Owner reconstructs
        let mut owner_rng = ChaCha20Rng::seed_from_u64(999);
        let (owner_secret, _owner_public) = generate_x25519_keypair(&mut owner_rng);
        let recon_result =
            reconstruct_shares(&owner_secret, &mut dist_result.worker_streams).await.unwrap();

        // Wait for worker tasks
        for handle in worker_handles {
            handle.await.unwrap();
        }

        // Verify reconstruction matches original
        assert_eq!(recon_result.weights.len(), weights.len());
        for (orig, recon) in weights.iter().zip(recon_result.weights.iter()) {
            assert!(
                (orig - recon).abs() < 1e-6,
                "mismatch: {} vs {}",
                orig,
                recon
            );
        }
    }

    #[tokio::test]
    async fn test_three_worker_network_with_simulated_training() {
        // 3 workers, simulate training by adding a gradient, then reconstruct.
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradient = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let shape = vec![5];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut worker_info = Vec::new();
        let mut worker_listeners = Vec::new();

        for i in 0..3 {
            let (secret, public) = generate_x25519_keypair(&mut rng);
            let party_id = PartyId::from_index(i);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            worker_info.push((party_id, addr, public, secret));
            worker_listeners.push(listener);
        }

        let owner_workers: Vec<(PartyId, SocketAddr, X25519PublicKey)> = worker_info
            .iter()
            .map(|(id, addr, pk, _)| (id.clone(), *addr, *pk))
            .collect();

        let gradient_clone = gradient.clone();

        // Spawn worker tasks
        let mut worker_handles = Vec::new();
        for (i, listener) in worker_listeners.into_iter().enumerate() {
            let party_id = worker_info[i].0.clone();
            let secret = worker_info[i].3.clone();
            let receiver = ShareReceiver::new(secret.clone(), party_id);
            let grad = gradient_clone.clone();

            let handle = tokio::spawn(async move {
                let (mut state, mut stream) =
                    worker_receive_distribution(&listener, &receiver, &secret).await.unwrap();

                // Simulate training: add gradient to share (only party 0 adds
                // the public gradient, maintaining the additive invariant).
                if state.weight_share.index == 0 {
                    for (j, g) in grad.iter().enumerate() {
                        let g_fr = Fr::from_f64(*g);
                        state.weight_share.data[j] =
                            Fr::add(&state.weight_share.data[j], &g_fr);
                    }
                }

                // Reconstruction: send modified share back
                worker_send_final_share(
                    &mut stream,
                    &state.weight_share,
                    &state.generators,
                    Some(700 + i as u64),
                )
                .await
                .unwrap();
            });
            worker_handles.push(handle);
        }

        // Owner distributes
        let mut dist_result =
            distribute_shares(&weights, &shape, &owner_workers, Some(200)).await.unwrap();
        assert!(dist_result.verified);

        // Owner reconstructs
        let mut owner_rng = ChaCha20Rng::seed_from_u64(888);
        let (owner_secret, _) = generate_x25519_keypair(&mut owner_rng);
        let recon_result =
            reconstruct_shares(&owner_secret, &mut dist_result.worker_streams).await.unwrap();

        // Wait for workers
        for handle in worker_handles {
            handle.await.unwrap();
        }

        // Verify: reconstructed = original + gradient
        let expected: Vec<f64> = weights
            .iter()
            .zip(gradient.iter())
            .map(|(w, g)| w + g)
            .collect();

        assert_eq!(recon_result.weights.len(), expected.len());
        for (exp, recon) in expected.iter().zip(recon_result.weights.iter()) {
            assert!(
                (exp - recon).abs() < 1e-6,
                "expected {}, got {}",
                exp,
                recon
            );
        }
    }
}
