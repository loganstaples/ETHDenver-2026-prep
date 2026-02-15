//! Share distribution, encrypted transport, Pedersen checkpoints, and weight reconstruction.
//!
//! This module implements the complete lifecycle for MPC weight management:
//!
//! 1. **Distribution** ([`ShareDistributor`]): Model owner splits weights into additive
//!    Fr shares and encrypts each share to the corresponding worker's x25519 public key.
//! 2. **Reception** ([`ShareReceiver`]): Workers decrypt their encrypted share using their
//!    x25519 private key, initializing their MPC state.
//! 3. **Checkpoints** ([`CheckpointCommitment`]): Workers independently compute Pedersen
//!    commitment shares over their weight shares. Any party can combine the commitment
//!    shares to verify the aggregate — without reconstructing the full weights.
//! 4. **Reconstruction** ([`WeightReconstructor`]): At training completion, workers encrypt
//!    their final shares to the model owner's public key. The owner collects, decrypts,
//!    sums to reconstruct the full trained weights, and verifies against the last
//!    checkpoint commitment.
//!
//! # Encryption scheme
//!
//! Share encryption uses ephemeral x25519 Diffie-Hellman key agreement followed by
//! AES-256-GCM authenticated encryption. This matches the pattern used throughout
//! helix-mpc (see `beaver::ot` and `session::secure_channel`).
//!
//! # Pedersen commitments
//!
//! Leverages the existing [`PedersenCommitment`] and [`PedersenGenerators`] from
//! `security::commitment`. The homomorphic property C(a,r1) + C(b,r2) = C(a+b, r1+r2)
//! allows combining per-worker commitment shares into a full commitment without any
//! party learning the underlying values.

use aes_gcm::{aead::Aead, Aes256Gcm, KeyInit, Nonce};
use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use x25519_dalek::StaticSecret;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::security::commitment::{PedersenCommitment, PedersenGenerators};
use crate::types::PartyId;

/// Re-export of x25519 public key type for downstream crates.
pub type X25519PublicKey = x25519_dalek::PublicKey;
/// Re-export of x25519 static secret type for downstream crates.
pub type X25519StaticSecret = x25519_dalek::StaticSecret;

// ============================================================================
// Data structures
// ============================================================================

/// An encrypted share ready for network transmission.
///
/// Contains the AES-256-GCM ciphertext of serialized Fr elements, plus
/// the sender's ephemeral public key needed for the recipient to derive
/// the decryption key via DH.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedShare {
    /// The recipient's party identifier.
    pub recipient: PartyId,
    /// The sender's ephemeral x25519 public key (32 bytes).
    pub ephemeral_public_key: [u8; 32],
    /// AES-256-GCM ciphertext of the serialized share data.
    pub ciphertext: Vec<u8>,
    /// Shape of the original weight tensor.
    pub shape: Vec<usize>,
    /// Number of Fr elements in the share.
    pub num_elements: usize,
}

/// A decrypted weight share held by a worker.
#[derive(Debug, Clone)]
pub struct WeightShare {
    /// The worker's party identifier.
    pub party: PartyId,
    /// Share index (0-based).
    pub index: usize,
    /// The additive share data as Fr field elements.
    pub data: Vec<Fr>,
    /// Shape of the original weight tensor.
    pub shape: Vec<usize>,
}

/// Result of the share distribution process.
pub struct DistributionResult {
    /// Encrypted shares, one per worker.
    pub encrypted_shares: Vec<EncryptedShare>,
    /// Initial Pedersen commitment to the full weights.
    /// This is the product of per-element commitments C_j = g^{w_j} * h^{r_j}.
    pub initial_commitment: VectorCommitment,
    /// Per-element blinding factors used for the initial commitment (owner keeps secret).
    pub blindings: Vec<Fr>,
}

/// A Pedersen commitment over a vector of Fr elements.
///
/// Contains per-element commitments and their aggregate (point sum).
/// The aggregate is suitable for on-chain storage; per-element commitments
/// enable fine-grained verification off-chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCommitment {
    /// Per-element Pedersen commitments: C_j = g^{v_j} * h^{r_j}.
    pub element_commitments: Vec<PedersenCommitment>,
    /// Aggregate = sum of all element commitment points.
    /// Due to homomorphism: aggregate = g^{sum(v_j)} * h^{sum(r_j)}.
    pub aggregate: PedersenCommitment,
}

impl VectorCommitment {
    /// Produces a deterministic 32-byte hash of the aggregate commitment.
    ///
    /// This is used for on-chain storage as a Solidity `bytes32` value.
    /// Delegates to `PedersenCommitment::to_bytes32()` on the aggregate point.
    pub fn to_bytes32(&self) -> [u8; 32] {
        self.aggregate.to_bytes32()
    }

    /// Verifies the full commitment against known values and blindings.
    pub fn verify(
        &self,
        values: &[Fr],
        blindings: &[Fr],
        generators: &PedersenGenerators,
    ) -> bool {
        if values.len() != self.element_commitments.len()
            || values.len() != blindings.len()
        {
            return false;
        }
        for (i, (v, r)) in values.iter().zip(blindings.iter()).enumerate() {
            if !self.element_commitments[i].verify(v, r, generators) {
                return false;
            }
        }
        true
    }

    /// Verifies only the aggregate commitment.
    ///
    /// Checks that aggregate == g^{sum(values)} * h^{sum(blindings)}.
    /// Faster than full element-wise verification.
    pub fn verify_aggregate(
        &self,
        values: &[Fr],
        total_blinding: &Fr,
        generators: &PedersenGenerators,
    ) -> bool {
        let sum: Fr = values.iter().fold(Fr::ZERO, |acc, v| Fr::add(&acc, v));
        self.aggregate.verify(&sum, total_blinding, generators)
    }
}

/// A single worker's Pedersen commitment share for a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentShare {
    /// The worker's party identifier.
    pub party: PartyId,
    /// Per-element commitment shares: C_i_j = g^{share_i_j} * h^{blinding_i_j}.
    pub element_commitments: Vec<PedersenCommitment>,
    /// Aggregate commitment share (sum of element commitments).
    pub aggregate: PedersenCommitment,
}

// ============================================================================
// ShareDistributor (model owner side)
// ============================================================================

/// Distributes model weights as encrypted additive shares to MPC workers.
///
/// The owner:
/// 1. Converts f64 weights to BN254 Fr field elements
/// 2. Generates N-1 random Fr shares; computes Nth as remainder
/// 3. Encrypts each share to the corresponding worker's x25519 public key
/// 4. Computes the initial Pedersen commitment C0 to the full weights
pub struct ShareDistributor {
    rng: ChaCha20Rng,
    generators: PedersenGenerators,
}

impl ShareDistributor {
    /// Creates a new ShareDistributor with entropy-seeded RNG.
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            generators: PedersenGenerators::default(),
        }
    }

    /// Creates a new ShareDistributor with a deterministic seed (for testing).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            generators: PedersenGenerators::default(),
        }
    }

    /// Creates a new ShareDistributor with custom Pedersen generators.
    pub fn with_generators(generators: PedersenGenerators) -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            generators,
        }
    }

    /// Returns a reference to the generators (workers need the same generators).
    pub fn generators(&self) -> &PedersenGenerators {
        &self.generators
    }

    /// Distributes weights to N workers.
    ///
    /// Converts `weights` (f64) to Fr field elements, generates additive shares,
    /// encrypts each share to the corresponding worker's x25519 public key, and
    /// computes the initial Pedersen commitment.
    pub fn distribute(
        &mut self,
        weights: &[f64],
        shape: &[usize],
        worker_public_keys: &[X25519PublicKey],
        worker_ids: &[PartyId],
    ) -> MPCResult<DistributionResult> {
        let n = worker_public_keys.len();
        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }
        if n != worker_ids.len() {
            return Err(MPCError::InvalidConfig(
                "worker_public_keys and worker_ids must have the same length".into(),
            ));
        }

        // Convert weights to Fr field elements
        let weight_frs: Vec<Fr> = weights.iter().map(|&w| Fr::from_f64(w)).collect();

        // Generate additive shares
        let shares = self.generate_additive_shares(&weight_frs, n);

        // Generate per-element blinding factors for the initial commitment
        let blindings: Vec<Fr> = (0..weights.len())
            .map(|_| Fr::random(&mut self.rng))
            .collect();

        // Compute initial Pedersen commitment to the full weights
        let initial_commitment = compute_vector_commitment(
            &weight_frs,
            &blindings,
            &self.generators,
        );

        // Encrypt each share to the corresponding worker's public key
        let mut encrypted_shares = Vec::with_capacity(n);
        for (i, share) in shares.iter().enumerate() {
            let enc = self.encrypt_share(share, &worker_public_keys[i], &worker_ids[i], shape)?;
            encrypted_shares.push(enc);
        }

        Ok(DistributionResult {
            encrypted_shares,
            initial_commitment,
            blindings,
        })
    }

    /// Distributes pre-computed Fr shares (for use when weights are already in Fr form).
    pub fn distribute_fr(
        &mut self,
        weight_frs: &[Fr],
        shape: &[usize],
        worker_public_keys: &[X25519PublicKey],
        worker_ids: &[PartyId],
    ) -> MPCResult<DistributionResult> {
        let n = worker_public_keys.len();
        if n < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: n,
            });
        }
        if n != worker_ids.len() {
            return Err(MPCError::InvalidConfig(
                "worker_public_keys and worker_ids must have the same length".into(),
            ));
        }

        let shares = self.generate_additive_shares(weight_frs, n);

        let blindings: Vec<Fr> = (0..weight_frs.len())
            .map(|_| Fr::random(&mut self.rng))
            .collect();

        let initial_commitment = compute_vector_commitment(
            weight_frs,
            &blindings,
            &self.generators,
        );

        let mut encrypted_shares = Vec::with_capacity(n);
        for (i, share) in shares.iter().enumerate() {
            let enc = self.encrypt_share(share, &worker_public_keys[i], &worker_ids[i], shape)?;
            encrypted_shares.push(enc);
        }

        Ok(DistributionResult {
            encrypted_shares,
            initial_commitment,
            blindings,
        })
    }

    /// Generates N additive shares of a vector of Fr elements.
    ///
    /// Shares 0..N-2 are uniformly random Fr elements.
    /// Share N-1 = value - sum(shares 0..N-2).
    pub(crate) fn generate_additive_shares(&mut self, values: &[Fr], n: usize) -> Vec<Vec<Fr>> {
        let dim = values.len();
        let mut shares: Vec<Vec<Fr>> = Vec::with_capacity(n);

        // Generate N-1 random shares
        for _ in 0..(n - 1) {
            let share: Vec<Fr> = (0..dim).map(|_| Fr::random(&mut self.rng)).collect();
            shares.push(share);
        }

        // Compute the Nth share as remainder: s_n[j] = value[j] - sum(s_i[j])
        let mut last_share = values.to_vec();
        for share in &shares {
            for (j, s) in share.iter().enumerate() {
                last_share[j] = Fr::sub(&last_share[j], s);
            }
        }
        shares.push(last_share);

        shares
    }

    /// Encrypts a share vector to a worker's x25519 public key.
    pub(crate) fn encrypt_share(
        &mut self,
        share: &[Fr],
        recipient_pk: &X25519PublicKey,
        recipient_id: &PartyId,
        shape: &[usize],
    ) -> MPCResult<EncryptedShare> {
        // Generate ephemeral x25519 key pair
        let mut ephemeral_bytes = [0u8; 32];
        self.rng.fill_bytes(&mut ephemeral_bytes);
        let ephemeral_secret = StaticSecret::from(ephemeral_bytes);
        let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

        // DH key agreement
        let shared_secret = ephemeral_secret.diffie_hellman(recipient_pk);

        // Derive AES key and nonce with domain separation
        let aes_key = derive_share_key(shared_secret.as_bytes(), ephemeral_public.as_bytes());
        let nonce = derive_share_nonce(shared_secret.as_bytes(), ephemeral_public.as_bytes());

        // Serialize Fr elements to bytes (32 bytes each, little-endian)
        let plaintext = serialize_fr_vec(share);

        // AES-256-GCM encrypt
        let ciphertext = encrypt_with_key(&aes_key, &plaintext, &nonce)?;

        Ok(EncryptedShare {
            recipient: recipient_id.clone(),
            ephemeral_public_key: *ephemeral_public.as_bytes(),
            ciphertext,
            shape: shape.to_vec(),
            num_elements: share.len(),
        })
    }
}

impl Default for ShareDistributor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ShareReceiver (worker side)
// ============================================================================

/// Receives and decrypts weight shares on the worker side.
///
/// Each worker holds an x25519 private key. When the owner distributes encrypted
/// shares, the worker uses [`ShareReceiver::receive`] to decrypt and initialize
/// their local MPC state.
pub struct ShareReceiver {
    secret_key: StaticSecret,
    party_id: PartyId,
}

impl ShareReceiver {
    /// Creates a new ShareReceiver with the worker's x25519 private key.
    pub fn new(secret_key: StaticSecret, party_id: PartyId) -> Self {
        Self {
            secret_key,
            party_id,
        }
    }

    /// Creates a new ShareReceiver from raw 32-byte key material.
    pub fn from_bytes(secret_bytes: [u8; 32], party_id: PartyId) -> Self {
        Self {
            secret_key: StaticSecret::from(secret_bytes),
            party_id,
        }
    }

    /// Returns the worker's x25519 public key (shared with the model owner).
    pub fn public_key(&self) -> X25519PublicKey {
        X25519PublicKey::from(&self.secret_key)
    }

    /// Returns a reference to the party ID.
    pub fn party_id(&self) -> &PartyId {
        &self.party_id
    }

    /// Decrypts an encrypted share and initializes the worker's weight share.
    ///
    /// Verifies that the encrypted share is addressed to this worker, performs
    /// DH key agreement with the sender's ephemeral public key, derives the
    /// AES key, and decrypts.
    pub fn receive(&self, encrypted: &EncryptedShare) -> MPCResult<WeightShare> {
        if encrypted.recipient != self.party_id {
            return Err(MPCError::ProtocolError(format!(
                "share addressed to {}, but this receiver is {}",
                encrypted.recipient, self.party_id
            )));
        }

        // Reconstruct the sender's ephemeral public key
        let ephemeral_pk = X25519PublicKey::from(encrypted.ephemeral_public_key);

        // DH key agreement
        let shared_secret = self.secret_key.diffie_hellman(&ephemeral_pk);

        // Derive AES key and nonce (same derivation as sender)
        let aes_key = derive_share_key(
            shared_secret.as_bytes(),
            &encrypted.ephemeral_public_key,
        );
        let nonce = derive_share_nonce(
            shared_secret.as_bytes(),
            &encrypted.ephemeral_public_key,
        );

        // Decrypt
        let plaintext = decrypt_with_key(&aes_key, &encrypted.ciphertext, &nonce)?;

        // Deserialize Fr elements
        let data = deserialize_fr_vec(&plaintext, encrypted.num_elements)?;

        let index = extract_party_index(&self.party_id);

        Ok(WeightShare {
            party: self.party_id.clone(),
            index,
            data,
            shape: encrypted.shape.clone(),
        })
    }
}

// ============================================================================
// CheckpointCommitment (worker + aggregator)
// ============================================================================

/// Handles Pedersen commitment checkpoints during training.
///
/// At each checkpoint:
/// 1. Each worker computes a commitment share over their local weight share
/// 2. Workers publish their commitment shares (G1 points — no secret data)
/// 3. Any party combines the shares into the full commitment
/// 4. The combined commitment goes on-chain as the checkpoint
///
/// Due to the homomorphic property of Pedersen commitments:
/// `C_1 + C_2 + ... + C_n = C(w_1 + w_2 + ... + w_n, r_1 + r_2 + ... + r_n)`
///
/// Nobody reconstructed the weights, yet the combined commitment binds the full model.
pub struct CheckpointCommitment {
    generators: PedersenGenerators,
}

impl CheckpointCommitment {
    /// Creates with default generators (HELIX-MPC-v1 nothing-up-my-sleeve).
    pub fn new() -> Self {
        Self {
            generators: PedersenGenerators::default(),
        }
    }

    /// Creates with custom generators (must match across all participants).
    pub fn with_generators(generators: PedersenGenerators) -> Self {
        Self { generators }
    }

    /// Returns a reference to the generators.
    pub fn generators(&self) -> &PedersenGenerators {
        &self.generators
    }

    /// Worker computes their Pedersen commitment share.
    ///
    /// Each worker calls this with their local weight share and fresh random
    /// blinding factors (one per element). Returns a [`CommitmentShare`] containing
    /// per-element commitment points and their aggregate.
    pub fn compute_share(
        &self,
        weight_share: &WeightShare,
        blindings: &[Fr],
    ) -> MPCResult<CommitmentShare> {
        if weight_share.data.len() != blindings.len() {
            return Err(MPCError::ShapeMismatch {
                expected: vec![weight_share.data.len()],
                got: vec![blindings.len()],
            });
        }

        let element_commitments: Vec<PedersenCommitment> = weight_share
            .data
            .iter()
            .zip(blindings.iter())
            .map(|(v, r)| PedersenCommitment::commit(v, r, &self.generators))
            .collect();

        let aggregate = aggregate_commitments(&element_commitments, &self.generators);

        Ok(CommitmentShare {
            party: weight_share.party.clone(),
            element_commitments,
            aggregate,
        })
    }

    /// Combines commitment shares from all workers into a full vector commitment.
    ///
    /// Due to homomorphism, the combined commitment is equivalent to a commitment
    /// to the full (reconstructed) weights with the sum of all blinding factors.
    /// This operation requires no secrets — any party or external observer can do it.
    pub fn combine_shares(
        &self,
        commitment_shares: &[CommitmentShare],
    ) -> MPCResult<VectorCommitment> {
        if commitment_shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 1,
                available: 0,
            });
        }

        let num_elements = commitment_shares[0].element_commitments.len();
        for cs in commitment_shares {
            if cs.element_commitments.len() != num_elements {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![num_elements],
                    got: vec![cs.element_commitments.len()],
                });
            }
        }

        // Combine per-element commitments: C_j = C_1_j + C_2_j + ... + C_n_j
        let mut combined_elements: Vec<PedersenCommitment> =
            commitment_shares[0].element_commitments.clone();
        for cs in &commitment_shares[1..] {
            for (i, c) in cs.element_commitments.iter().enumerate() {
                combined_elements[i] = combined_elements[i].add(c);
            }
        }

        // Combine aggregate commitments
        let mut combined_aggregate = commitment_shares[0].aggregate.clone();
        for cs in &commitment_shares[1..] {
            combined_aggregate = combined_aggregate.add(&cs.aggregate);
        }

        Ok(VectorCommitment {
            element_commitments: combined_elements,
            aggregate: combined_aggregate,
        })
    }

    /// Verifies a combined commitment against known values and blindings.
    ///
    /// Used by the model owner at reconstruction time to verify that the
    /// reconstructed weights match the last checkpoint commitment.
    pub fn verify_commitment(
        &self,
        commitment: &VectorCommitment,
        values: &[Fr],
        blindings: &[Fr],
    ) -> bool {
        commitment.verify(values, blindings, &self.generators)
    }
}

impl Default for CheckpointCommitment {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WeightReconstructor (model owner side)
// ============================================================================

/// Reconstructs the full trained weights from worker shares at training completion.
///
/// The owner:
/// 1. Receives encrypted final shares from each worker
/// 2. Decrypts using their x25519 private key
/// 3. Sums all shares to reconstruct the full weights
/// 4. Optionally verifies against the last checkpoint commitment
/// 5. Converts Fr elements back to f64
pub struct WeightReconstructor {
    secret_key: StaticSecret,
    generators: PedersenGenerators,
}

impl WeightReconstructor {
    /// Creates with the owner's x25519 private key.
    pub fn new(secret_key: StaticSecret) -> Self {
        Self {
            secret_key,
            generators: PedersenGenerators::default(),
        }
    }

    /// Creates from raw 32-byte key material.
    pub fn from_bytes(secret_bytes: [u8; 32]) -> Self {
        Self {
            secret_key: StaticSecret::from(secret_bytes),
            generators: PedersenGenerators::default(),
        }
    }

    /// Returns the owner's public key (workers encrypt their final shares to this).
    pub fn public_key(&self) -> X25519PublicKey {
        X25519PublicKey::from(&self.secret_key)
    }

    /// Decrypts all shares, sums to reconstruct weights, optionally verifies checkpoint.
    ///
    /// # Arguments
    /// * `encrypted_shares` - One encrypted share from each worker
    /// * `checkpoint` - Last checkpoint commitment for verification (optional)
    /// * `checkpoint_blindings` - Combined blinding factors from all workers (optional)
    ///
    /// If `checkpoint` and `checkpoint_blindings` are provided, verifies the
    /// reconstructed weights match the checkpoint commitment. Returns an error
    /// if verification fails.
    pub fn reconstruct(
        &self,
        encrypted_shares: &[EncryptedShare],
        checkpoint: Option<&VectorCommitment>,
        checkpoint_blindings: Option<&[Fr]>,
    ) -> MPCResult<Vec<f64>> {
        if encrypted_shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let num_elements = encrypted_shares[0].num_elements;

        // Decrypt all shares
        let mut shares: Vec<Vec<Fr>> = Vec::with_capacity(encrypted_shares.len());
        for enc in encrypted_shares {
            if enc.num_elements != num_elements {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![num_elements],
                    got: vec![enc.num_elements],
                });
            }

            let ephemeral_pk = X25519PublicKey::from(enc.ephemeral_public_key);
            let shared_secret = self.secret_key.diffie_hellman(&ephemeral_pk);

            let aes_key = derive_share_key(
                shared_secret.as_bytes(),
                &enc.ephemeral_public_key,
            );
            let nonce = derive_share_nonce(
                shared_secret.as_bytes(),
                &enc.ephemeral_public_key,
            );

            let plaintext = decrypt_with_key(&aes_key, &enc.ciphertext, &nonce)?;
            let data = deserialize_fr_vec(&plaintext, num_elements)?;
            shares.push(data);
        }

        // Sum all shares to reconstruct: w[j] = sum_i(share_i[j])
        let mut reconstructed = vec![Fr::ZERO; num_elements];
        for share in &shares {
            for (j, s) in share.iter().enumerate() {
                reconstructed[j] = Fr::add(&reconstructed[j], s);
            }
        }

        // Verify against checkpoint if provided
        if let (Some(commitment), Some(blindings)) = (checkpoint, checkpoint_blindings) {
            if blindings.len() != num_elements {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![num_elements],
                    got: vec![blindings.len()],
                });
            }
            if !commitment.verify(&reconstructed, blindings, &self.generators) {
                return Err(MPCError::CommitmentVerificationFailed {
                    party: PartyId::new("owner"),
                });
            }
        }

        // Convert back to f64
        let weights: Vec<f64> = reconstructed.iter().map(|fr| fr.to_f64()).collect();
        Ok(weights)
    }

    /// Reconstructs from decrypted WeightShares directly (no encryption layer).
    ///
    /// Useful when shares have already been decrypted or in testing.
    pub fn reconstruct_from_shares(
        &self,
        shares: &[WeightShare],
        checkpoint: Option<&VectorCommitment>,
        checkpoint_blindings: Option<&[Fr]>,
    ) -> MPCResult<Vec<f64>> {
        if shares.is_empty() {
            return Err(MPCError::InsufficientShares {
                required: 2,
                available: 0,
            });
        }

        let num_elements = shares[0].data.len();
        for s in shares {
            if s.data.len() != num_elements {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![num_elements],
                    got: vec![s.data.len()],
                });
            }
        }

        let mut reconstructed = vec![Fr::ZERO; num_elements];
        for share in shares {
            for (j, s) in share.data.iter().enumerate() {
                reconstructed[j] = Fr::add(&reconstructed[j], s);
            }
        }

        if let (Some(commitment), Some(blindings)) = (checkpoint, checkpoint_blindings) {
            if blindings.len() != num_elements {
                return Err(MPCError::ShapeMismatch {
                    expected: vec![num_elements],
                    got: vec![blindings.len()],
                });
            }
            if !commitment.verify(&reconstructed, blindings, &self.generators) {
                return Err(MPCError::CommitmentVerificationFailed {
                    party: PartyId::new("owner"),
                });
            }
        }

        let weights: Vec<f64> = reconstructed.iter().map(|fr| fr.to_f64()).collect();
        Ok(weights)
    }
}

// ============================================================================
// Standalone helper: encrypt a share for the owner
// ============================================================================

/// Encrypts a worker's final share to the model owner's x25519 public key.
///
/// Workers call this at training completion to send their trained weight share
/// back to the model owner for reconstruction.
pub fn encrypt_share_for_owner(
    share: &WeightShare,
    owner_public_key: &X25519PublicKey,
    rng: &mut impl RngCore,
) -> MPCResult<EncryptedShare> {
    let mut ephemeral_bytes = [0u8; 32];
    rng.fill_bytes(&mut ephemeral_bytes);
    let ephemeral_secret = StaticSecret::from(ephemeral_bytes);
    let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

    let shared_secret = ephemeral_secret.diffie_hellman(owner_public_key);

    let aes_key = derive_share_key(shared_secret.as_bytes(), ephemeral_public.as_bytes());
    let nonce = derive_share_nonce(shared_secret.as_bytes(), ephemeral_public.as_bytes());

    let plaintext = serialize_fr_vec(&share.data);
    let ciphertext = encrypt_with_key(&aes_key, &plaintext, &nonce)?;

    Ok(EncryptedShare {
        recipient: PartyId::new("owner"),
        ephemeral_public_key: *ephemeral_public.as_bytes(),
        ciphertext,
        shape: share.shape.clone(),
        num_elements: share.data.len(),
    })
}

// ============================================================================
// Key generation
// ============================================================================

/// Generates an x25519 key pair for MPC share distribution.
pub fn generate_x25519_keypair(rng: &mut impl RngCore) -> (StaticSecret, X25519PublicKey) {
    let mut secret_bytes = [0u8; 32];
    rng.fill_bytes(&mut secret_bytes);
    let secret = StaticSecret::from(secret_bytes);
    let public = X25519PublicKey::from(&secret);
    (secret, public)
}

// ============================================================================
// Internal: vector commitment helper
// ============================================================================

pub(crate) fn compute_vector_commitment(
    values: &[Fr],
    blindings: &[Fr],
    generators: &PedersenGenerators,
) -> VectorCommitment {
    let element_commitments: Vec<PedersenCommitment> = values
        .iter()
        .zip(blindings.iter())
        .map(|(v, r)| PedersenCommitment::commit(v, r, generators))
        .collect();

    let aggregate = aggregate_commitments(&element_commitments, generators);

    VectorCommitment {
        element_commitments,
        aggregate,
    }
}

fn aggregate_commitments(
    commitments: &[PedersenCommitment],
    generators: &PedersenGenerators,
) -> PedersenCommitment {
    if commitments.is_empty() {
        return PedersenCommitment::commit(&Fr::ZERO, &Fr::ZERO, generators);
    }
    let mut agg = commitments[0].clone();
    for c in &commitments[1..] {
        agg = agg.add(c);
    }
    agg
}

// ============================================================================
// Internal: crypto helpers (following patterns from beaver/ot.rs)
// ============================================================================

/// Derives an AES-256 key from the DH shared secret for share encryption.
/// Uses domain-separated SHA-256 to ensure independence from other key derivations.
pub(crate) fn derive_share_key(shared_secret: &[u8; 32], ephemeral_pk: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX-SHARE-KEY-v1");
    hasher.update(shared_secret);
    hasher.update(ephemeral_pk);
    hasher.finalize().into()
}

/// Derives a 96-bit nonce for AES-GCM from the DH shared secret.
pub(crate) fn derive_share_nonce(shared_secret: &[u8; 32], ephemeral_pk: &[u8; 32]) -> [u8; 12] {
    let mut hasher = Sha256::new();
    hasher.update(b"HELIX-SHARE-NONCE-v1");
    hasher.update(shared_secret);
    hasher.update(ephemeral_pk);
    let hash: [u8; 32] = hasher.finalize().into();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&hash[..12]);
    nonce
}

/// Encrypts plaintext using AES-256-GCM.
pub(crate) fn encrypt_with_key(key: &[u8; 32], plaintext: &[u8], nonce: &[u8; 12]) -> MPCResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| MPCError::ProtocolError(format!("cipher init failed: {}", e)))?;
    let nonce = Nonce::from_slice(nonce);
    cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| MPCError::ProtocolError(format!("encryption failed: {}", e)))
}

/// Decrypts ciphertext using AES-256-GCM.
pub(crate) fn decrypt_with_key(key: &[u8; 32], ciphertext: &[u8], nonce: &[u8; 12]) -> MPCResult<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| MPCError::ProtocolError(format!("cipher init failed: {}", e)))?;
    let nonce = Nonce::from_slice(nonce);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| MPCError::ProtocolError(format!("decryption failed: {}", e)))
}

/// Serializes a vector of Fr elements to bytes (32 bytes per element, little-endian).
pub(crate) fn serialize_fr_vec(values: &[Fr]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 32);
    for v in values {
        data.extend_from_slice(&v.to_bytes_le());
    }
    data
}

/// Deserializes a byte vector into Fr elements.
pub(crate) fn deserialize_fr_vec(data: &[u8], expected_count: usize) -> MPCResult<Vec<Fr>> {
    let expected_bytes = expected_count * 32;
    if data.len() != expected_bytes {
        return Err(MPCError::ProtocolError(format!(
            "expected {} bytes for {} Fr elements, got {}",
            expected_bytes, expected_count, data.len()
        )));
    }

    let mut values = Vec::with_capacity(expected_count);
    for i in 0..expected_count {
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&data[i * 32..(i + 1) * 32]);
        values.push(Fr::from_bytes_le(&bytes));
    }
    Ok(values)
}

/// Extracts the numeric index from a PartyId like "party-2".
fn extract_party_index(party_id: &PartyId) -> usize {
    party_id
        .0
        .strip_prefix("party-")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: creates N workers with x25519 key pairs and party IDs.
    fn setup_workers(
        n: usize,
        seed: u64,
    ) -> Vec<(StaticSecret, X25519PublicKey, PartyId)> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let (secret, public) = generate_x25519_keypair(&mut rng);
                let party_id = PartyId::from_index(i);
                (secret, public, party_id)
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // test_share_roundtrip
    // ------------------------------------------------------------------

    #[test]
    fn test_share_roundtrip() {
        // Owner splits weights into 3 shares, workers receive shares,
        // workers send shares back, owner reconstructs and verifies.
        let weights = vec![0.1, -0.5, 1.2, 0.0, -3.14, 2.718];
        let shape = vec![2, 3];

        // Setup 3 workers
        let workers = setup_workers(3, 42);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        // Owner distributes
        let mut distributor = ShareDistributor::with_seed(100);
        let result = distributor
            .distribute(&weights, &shape, &worker_pks, &worker_ids)
            .expect("distribution should succeed");
        assert_eq!(result.encrypted_shares.len(), 3);

        // Each worker receives their share
        let mut received_shares = Vec::new();
        for (i, (secret, _, party_id)) in workers.iter().enumerate() {
            let receiver = ShareReceiver::new(secret.clone(), party_id.clone());
            let share = receiver
                .receive(&result.encrypted_shares[i])
                .expect("reception should succeed");
            assert_eq!(share.data.len(), weights.len());
            assert_eq!(share.shape, shape);
            received_shares.push(share);
        }

        // Workers send shares back to owner for reconstruction
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        let (owner_secret, owner_pk) = generate_x25519_keypair(&mut rng);

        let mut encrypted_back: Vec<EncryptedShare> = Vec::new();
        for share in &received_shares {
            let enc = encrypt_share_for_owner(share, &owner_pk, &mut rng)
                .expect("encrypt for owner should succeed");
            encrypted_back.push(enc);
        }

        // Owner reconstructs
        let reconstructor = WeightReconstructor::new(owner_secret);
        let reconstructed = reconstructor
            .reconstruct(&encrypted_back, None, None)
            .expect("reconstruction should succeed");

        // Verify reconstruction matches original
        assert_eq!(reconstructed.len(), weights.len());
        for (orig, recon) in weights.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-6,
                "weight mismatch: original={}, reconstructed={}",
                orig,
                recon
            );
        }
    }

    // ------------------------------------------------------------------
    // test_pedersen_commitment_verification
    // ------------------------------------------------------------------

    #[test]
    fn test_pedersen_commitment_verification() {
        // Split weights, compute per-party commitment shares, combine,
        // verify against full commitment.
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        let workers = setup_workers(3, 42);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        // Distribute
        let mut distributor = ShareDistributor::with_seed(100);
        let result = distributor
            .distribute(&weights, &shape, &worker_pks, &worker_ids)
            .expect("distribution should succeed");

        // Workers receive shares
        let mut received_shares = Vec::new();
        for (i, (secret, _, party_id)) in workers.iter().enumerate() {
            let receiver = ShareReceiver::new(secret.clone(), party_id.clone());
            let share = receiver.receive(&result.encrypted_shares[i]).unwrap();
            received_shares.push(share);
        }

        // Each worker computes their commitment share with their own blinding
        let checkpoint = CheckpointCommitment::new();
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let mut all_blindings: Vec<Vec<Fr>> = Vec::new();
        let mut commitment_shares = Vec::new();

        for share in &received_shares {
            let blindings: Vec<Fr> = (0..share.data.len())
                .map(|_| Fr::random(&mut rng))
                .collect();
            let cs = checkpoint
                .compute_share(share, &blindings)
                .expect("compute share should succeed");
            commitment_shares.push(cs);
            all_blindings.push(blindings);
        }

        // Combine commitment shares
        let combined = checkpoint
            .combine_shares(&commitment_shares)
            .expect("combine should succeed");

        // Compute total blinding per element (sum across workers)
        let total_blindings: Vec<Fr> = (0..weights.len())
            .map(|j| {
                let mut sum = Fr::ZERO;
                for blindings in &all_blindings {
                    sum = Fr::add(&sum, &blindings[j]);
                }
                sum
            })
            .collect();

        // Verify: combined commitment should match the full weights
        let weight_frs: Vec<Fr> = weights.iter().map(|&w| Fr::from_f64(w)).collect();
        assert!(
            checkpoint.verify_commitment(&combined, &weight_frs, &total_blindings),
            "combined commitment should verify against full weights"
        );

        // Negative test: wrong values should fail
        let wrong_weights: Vec<Fr> = vec![
            Fr::from_f64(1.0),
            Fr::from_f64(2.0),
            Fr::from_f64(3.0),
            Fr::from_f64(5.0), // wrong!
        ];
        assert!(
            !checkpoint.verify_commitment(&combined, &wrong_weights, &total_blindings),
            "commitment should NOT verify against wrong weights"
        );
    }

    // ------------------------------------------------------------------
    // test_commitment_without_reconstruction
    // ------------------------------------------------------------------

    #[test]
    fn test_commitment_without_reconstruction() {
        // 3 workers each compute their commitment share, combine them,
        // verify the combined commitment matches — without any party
        // seeing the full weights.
        let weights = vec![10.0, 20.0, 30.0];
        let weight_frs: Vec<Fr> = weights.iter().map(|&w| Fr::from_f64(w)).collect();

        let n = 3;
        let dim = weights.len();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Simulate additive sharing directly (no encryption in this test)
        let mut shares: Vec<Vec<Fr>> = Vec::new();
        for _ in 0..(n - 1) {
            let share: Vec<Fr> = (0..dim).map(|_| Fr::random(&mut rng)).collect();
            shares.push(share);
        }
        let mut last = weight_frs.clone();
        for share in &shares {
            for (j, s) in share.iter().enumerate() {
                last[j] = Fr::sub(&last[j], s);
            }
        }
        shares.push(last);

        // Verify that individual shares don't reveal the weights
        // (each random share is information-theoretically independent of the secret)
        for (i, share) in shares.iter().enumerate().take(n - 1) {
            let share_f64: Vec<f64> = share.iter().map(|s| s.to_f64()).collect();
            let matches_original = share_f64
                .iter()
                .zip(weights.iter())
                .all(|(s, w)| (s - w).abs() < 1e-6);
            assert!(
                !matches_original,
                "share {} should not reveal original weights",
                i
            );
        }

        // Each worker independently computes their Pedersen commitment share
        let checkpoint = CheckpointCommitment::new();
        let mut all_blindings: Vec<Vec<Fr>> = Vec::new();
        let mut commitment_shares = Vec::new();

        for (i, share) in shares.iter().enumerate() {
            let blindings: Vec<Fr> = (0..dim).map(|_| Fr::random(&mut rng)).collect();
            let weight_share = WeightShare {
                party: PartyId::from_index(i),
                index: i,
                data: share.clone(),
                shape: vec![dim],
            };
            let cs = checkpoint.compute_share(&weight_share, &blindings).unwrap();
            commitment_shares.push(cs);
            all_blindings.push(blindings);
        }

        // Anyone can combine the commitment shares (no secrets needed)
        let combined = checkpoint.combine_shares(&commitment_shares).unwrap();

        // Verify: combined commitment matches C(full_weights, total_blinding)
        let total_blindings: Vec<Fr> = (0..dim)
            .map(|j| {
                let mut sum = Fr::ZERO;
                for blindings in &all_blindings {
                    sum = Fr::add(&sum, &blindings[j]);
                }
                sum
            })
            .collect();

        assert!(
            checkpoint.verify_commitment(&combined, &weight_frs, &total_blindings),
            "combined commitment should verify without any party seeing full weights"
        );

        // Also verify the aggregate commitment
        let total_blinding_sum: Fr = total_blindings
            .iter()
            .fold(Fr::ZERO, |acc, r| Fr::add(&acc, r));
        assert!(
            combined.verify_aggregate(&weight_frs, &total_blinding_sum, &checkpoint.generators),
            "aggregate commitment should verify"
        );
    }

    // ------------------------------------------------------------------
    // test_encrypted_share_transfer
    // ------------------------------------------------------------------

    #[test]
    fn test_encrypted_share_transfer() {
        // Encrypt a share to a worker's key, have the worker decrypt, verify correctness.
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Generate worker key pair
        let (worker_secret, worker_pk) = generate_x25519_keypair(&mut rng);
        let worker_id = PartyId::from_index(0);

        // Create a share
        let share_data: Vec<Fr> = vec![
            Fr::from_f64(0.1),
            Fr::from_f64(-0.5),
            Fr::from_f64(1.0),
            Fr::from_f64(0.0),
            Fr::from_f64(-999.123456),
        ];

        // Owner encrypts to worker
        let mut distributor = ShareDistributor::with_seed(100);
        let encrypted = distributor
            .encrypt_share(&share_data, &worker_pk, &worker_id, &[5])
            .expect("encryption should succeed");

        assert_eq!(encrypted.recipient, worker_id);
        assert_eq!(encrypted.num_elements, 5);
        assert_eq!(encrypted.shape, vec![5]);

        // Ciphertext should be larger than plaintext (includes auth tag)
        assert!(encrypted.ciphertext.len() > share_data.len() * 32);

        // Worker decrypts
        let receiver = ShareReceiver::new(worker_secret, worker_id.clone());
        let received = receiver.receive(&encrypted).expect("decryption should succeed");

        // Verify the decrypted data exactly matches
        assert_eq!(received.data.len(), share_data.len());
        for (original, decrypted) in share_data.iter().zip(received.data.iter()) {
            assert!(
                original.ct_eq(decrypted).to_bool(),
                "decrypted share should exactly match original"
            );
        }

        // Wrong worker should fail
        let (wrong_secret, _) = generate_x25519_keypair(&mut rng);
        let wrong_receiver = ShareReceiver::new(wrong_secret, worker_id);
        assert!(
            wrong_receiver.receive(&encrypted).is_err(),
            "decryption with wrong key should fail"
        );
    }

    // ------------------------------------------------------------------
    // test_large_model
    // ------------------------------------------------------------------

    #[test]
    fn test_large_model() {
        // Full flow with a 25K-param model (MNIST 784->32->10).
        // 784*32 + 32 + 32*10 + 10 = 25,450 parameters.
        let total_params = 784 * 32 + 32 + 32 * 10 + 10;
        assert_eq!(total_params, 25450);

        // Generate realistic Xavier-initialized weights
        let weights: Vec<f64> = (0..total_params)
            .map(|i| {
                let scale = if i < 784 * 32 {
                    (2.0 / 784.0_f64).sqrt()
                } else if i < 784 * 32 + 32 {
                    0.0 // bias init
                } else if i < 784 * 32 + 32 + 32 * 10 {
                    (2.0 / 32.0_f64).sqrt()
                } else {
                    0.0 // bias init
                };
                // Deterministic pseudo-random initialization
                let t = (i as f64) * 0.618033988749895; // golden ratio
                let pseudo = t - t.floor() - 0.5;
                pseudo * 2.0 * scale
            })
            .collect();

        let shape = vec![total_params];

        // Setup 3 workers
        let workers = setup_workers(3, 100);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        // === Phase 1: Distribution ===
        let mut distributor = ShareDistributor::with_seed(200);
        let result = distributor
            .distribute(&weights, &shape, &worker_pks, &worker_ids)
            .expect("distribution of 25K params should succeed");

        assert_eq!(result.encrypted_shares.len(), 3);
        assert_eq!(result.initial_commitment.element_commitments.len(), total_params);

        // === Phase 2: Workers receive shares ===
        let mut received_shares = Vec::new();
        for (i, (secret, _, party_id)) in workers.iter().enumerate() {
            let receiver = ShareReceiver::new(secret.clone(), party_id.clone());
            let share = receiver.receive(&result.encrypted_shares[i]).unwrap();
            assert_eq!(share.data.len(), total_params);
            received_shares.push(share);
        }

        // === Phase 3: Checkpoint commitment ===
        let checkpoint = CheckpointCommitment::new();
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        let mut all_blindings: Vec<Vec<Fr>> = Vec::new();
        let mut commitment_shares = Vec::new();

        for share in &received_shares {
            let blindings: Vec<Fr> = (0..share.data.len())
                .map(|_| Fr::random(&mut rng))
                .collect();
            let cs = checkpoint.compute_share(share, &blindings).unwrap();
            commitment_shares.push(cs);
            all_blindings.push(blindings);
        }

        let combined = checkpoint.combine_shares(&commitment_shares).unwrap();
        assert_eq!(combined.element_commitments.len(), total_params);

        // === Phase 4: Workers send shares to owner ===
        let (owner_secret, owner_pk) = generate_x25519_keypair(&mut rng);
        let mut encrypted_back = Vec::new();
        for share in &received_shares {
            let enc = encrypt_share_for_owner(share, &owner_pk, &mut rng).unwrap();
            encrypted_back.push(enc);
        }

        // === Phase 5: Owner reconstructs with checkpoint verification ===
        // Compute total blindings for checkpoint verification
        let total_blindings: Vec<Fr> = (0..total_params)
            .map(|j| {
                let mut sum = Fr::ZERO;
                for blindings in &all_blindings {
                    sum = Fr::add(&sum, &blindings[j]);
                }
                sum
            })
            .collect();

        let reconstructor = WeightReconstructor::new(owner_secret);
        let reconstructed = reconstructor
            .reconstruct(
                &encrypted_back,
                Some(&combined),
                Some(&total_blindings),
            )
            .expect("reconstruction with checkpoint verification should succeed");

        // Verify reconstruction matches original
        assert_eq!(reconstructed.len(), weights.len());
        let mut max_error = 0.0f64;
        for (orig, recon) in weights.iter().zip(reconstructed.iter()) {
            let error = (orig - recon).abs();
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-6,
            "max reconstruction error {} should be < 1e-6",
            max_error
        );
    }

    // ------------------------------------------------------------------
    // Additional edge-case tests
    // ------------------------------------------------------------------

    #[test]
    fn test_two_party_sharing() {
        // Minimal: 2 workers.
        let weights = vec![42.0, -17.5];
        let shape = vec![2];

        let workers = setup_workers(2, 42);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        let mut distributor = ShareDistributor::with_seed(100);
        let result = distributor.distribute(&weights, &shape, &worker_pks, &worker_ids).unwrap();

        let mut received = Vec::new();
        for (i, (secret, _, party_id)) in workers.iter().enumerate() {
            let receiver = ShareReceiver::new(secret.clone(), party_id.clone());
            received.push(receiver.receive(&result.encrypted_shares[i]).unwrap());
        }

        // Reconstruct without encryption (direct from shares)
        let reconstructor = WeightReconstructor::from_bytes([0u8; 32]);
        let reconstructed = reconstructor
            .reconstruct_from_shares(&received, None, None)
            .unwrap();

        for (orig, recon) in weights.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - recon).abs() < 1e-6,
                "2-party roundtrip mismatch: {} vs {}",
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_insufficient_parties_error() {
        let mut distributor = ShareDistributor::with_seed(42);
        let workers = setup_workers(1, 42);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        let result = distributor.distribute(&[1.0], &[1], &worker_pks, &worker_ids);
        assert!(result.is_err(), "should require at least 2 parties");
    }

    #[test]
    fn test_wrong_recipient_error() {
        let workers = setup_workers(2, 42);
        let worker_pks: Vec<X25519PublicKey> =
            workers.iter().map(|(_, pk, _)| *pk).collect();
        let worker_ids: Vec<PartyId> =
            workers.iter().map(|(_, _, id)| id.clone()).collect();

        let mut distributor = ShareDistributor::with_seed(100);
        let result = distributor
            .distribute(&[1.0, 2.0], &[2], &worker_pks, &worker_ids)
            .unwrap();

        // Worker 1 tries to receive worker 0's share
        let receiver = ShareReceiver::new(workers[1].0.clone(), workers[1].2.clone());
        let err = receiver.receive(&result.encrypted_shares[0]);
        assert!(err.is_err(), "wrong recipient should fail");
    }

    #[test]
    fn test_additive_shares_sum_to_original() {
        // Verify the fundamental property: sum of all shares = original value
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let values: Vec<Fr> = (0..100).map(|_| Fr::random(&mut rng)).collect();

        let mut distributor = ShareDistributor::with_seed(100);
        let shares = distributor.generate_additive_shares(&values, 5);

        assert_eq!(shares.len(), 5);
        for share in &shares {
            assert_eq!(share.len(), 100);
        }

        // Sum all shares element-wise
        let mut reconstructed = vec![Fr::ZERO; 100];
        for share in &shares {
            for (j, s) in share.iter().enumerate() {
                reconstructed[j] = Fr::add(&reconstructed[j], s);
            }
        }

        // Must exactly match (field arithmetic is exact, no floating point)
        for (orig, recon) in values.iter().zip(reconstructed.iter()) {
            assert!(
                orig.ct_eq(recon).to_bool(),
                "additive shares must sum to exact original value"
            );
        }
    }

    #[test]
    fn test_checkpoint_commitment_with_modified_weights() {
        // Simulate a training step that modifies weights, then verify
        // the checkpoint commitment reflects the new weights.
        let n = 3;
        let dim = 10;
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Initial shares (simulated)
        let initial_weights: Vec<Fr> = (0..dim).map(|_| Fr::random(&mut rng)).collect();
        let mut distributor = ShareDistributor::with_seed(100);
        let shares = distributor.generate_additive_shares(&initial_weights, n);

        // Simulate: each worker applies a gradient update to their share
        // In MPC, this is done locally without communication for addition
        let gradient: Vec<Fr> = (0..dim).map(|i| Fr::from_f64(0.01 * (i as f64))).collect();
        let updated_shares: Vec<Vec<Fr>> = shares
            .iter()
            .enumerate()
            .map(|(party_idx, share)| {
                share
                    .iter()
                    .zip(gradient.iter())
                    .map(|(s, g)| {
                        // Only party 0 adds the public gradient
                        if party_idx == 0 {
                            Fr::add(s, g)
                        } else {
                            *s
                        }
                    })
                    .collect()
            })
            .collect();

        // Workers compute checkpoint commitments on updated shares
        let checkpoint = CheckpointCommitment::new();
        let mut all_blindings: Vec<Vec<Fr>> = Vec::new();
        let mut commitment_shares = Vec::new();

        for (i, share) in updated_shares.iter().enumerate() {
            let blindings: Vec<Fr> = (0..dim).map(|_| Fr::random(&mut rng)).collect();
            let ws = WeightShare {
                party: PartyId::from_index(i),
                index: i,
                data: share.clone(),
                shape: vec![dim],
            };
            let cs = checkpoint.compute_share(&ws, &blindings).unwrap();
            commitment_shares.push(cs);
            all_blindings.push(blindings);
        }

        let combined = checkpoint.combine_shares(&commitment_shares).unwrap();

        // Expected updated weights = initial + gradient
        let expected: Vec<Fr> = initial_weights
            .iter()
            .zip(gradient.iter())
            .map(|(w, g)| Fr::add(w, g))
            .collect();

        let total_blindings: Vec<Fr> = (0..dim)
            .map(|j| {
                let mut sum = Fr::ZERO;
                for blindings in &all_blindings {
                    sum = Fr::add(&sum, &blindings[j]);
                }
                sum
            })
            .collect();

        assert!(
            checkpoint.verify_commitment(&combined, &expected, &total_blindings),
            "checkpoint should verify with updated weights"
        );

        // Old weights should NOT verify
        assert!(
            !checkpoint.verify_commitment(&combined, &initial_weights, &total_blindings),
            "checkpoint should NOT verify with old weights"
        );
    }

    #[test]
    fn test_serialization_roundtrip() {
        // Verify Fr serialization/deserialization is exact
        let values: Vec<Fr> = vec![
            Fr::ZERO,
            Fr::ONE,
            Fr::from_f64(3.14159),
            Fr::from_f64(-2.71828),
            Fr::from_u64(u64::MAX),
        ];

        let bytes = serialize_fr_vec(&values);
        assert_eq!(bytes.len(), values.len() * 32);

        let recovered = deserialize_fr_vec(&bytes, values.len()).unwrap();
        assert_eq!(recovered.len(), values.len());

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            assert!(
                orig.ct_eq(recov).to_bool(),
                "serialization roundtrip should be exact"
            );
        }
    }

    #[test]
    fn test_vector_commitment_aggregate_verification() {
        // Test that the aggregate commitment properly represents the sum
        let generators = PedersenGenerators::default();
        let values = vec![
            Fr::from_f64(1.0),
            Fr::from_f64(2.0),
            Fr::from_f64(3.0),
        ];
        let blindings = vec![
            Fr::from_f64(10.0),
            Fr::from_f64(20.0),
            Fr::from_f64(30.0),
        ];

        let vc = compute_vector_commitment(&values, &blindings, &generators);

        // Aggregate should verify with sum of values and sum of blindings
        let value_sum = Fr::from_f64(6.0);
        let blinding_sum = Fr::from_f64(60.0);
        assert!(
            vc.verify_aggregate(&values, &blinding_sum, &generators),
            "aggregate should verify with sum of blindings"
        );

        // Direct verification of the aggregate point
        assert!(
            vc.aggregate.verify(&value_sum, &blinding_sum, &generators),
            "aggregate point should match commitment to sums"
        );
    }
}
