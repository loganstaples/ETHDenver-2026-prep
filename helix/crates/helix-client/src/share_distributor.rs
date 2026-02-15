//! Model owner share distribution and weight reconstruction.
//!
//! This module provides the client-side interface for the MPC weight lifecycle:
//!
//! - **[`ModelOwnerDistributor`]**: High-level orchestrator that manages the complete
//!   weight distribution flow — splitting, encrypting, distributing shares, and
//!   computing the initial Pedersen commitment.
//! - **Weight reconstruction**: Collecting encrypted final shares from workers and
//!   reconstructing the trained model weights with checkpoint verification.
//!
//! The underlying cryptographic operations are provided by `helix_mpc::share_distribution`.

pub use helix_mpc::share_distribution::{
    CheckpointCommitment, CommitmentShare, DistributionResult, EncryptedShare,
    ShareDistributor, ShareReceiver, VectorCommitment, WeightReconstructor, WeightShare,
    X25519PublicKey, X25519StaticSecret,
    encrypt_share_for_owner, generate_x25519_keypair,
};
pub use helix_mpc::security::commitment::{PedersenCommitment, PedersenGenerators};
pub use helix_mpc::{Fr, PartyId};

/// High-level model owner operations for MPC weight distribution and reconstruction.
///
/// Wraps [`ShareDistributor`] and [`WeightReconstructor`] into a single interface
/// that manages the owner's x25519 key pair and Pedersen generators.
pub struct ModelOwnerDistributor {
    distributor: ShareDistributor,
    owner_secret: X25519StaticSecret,
    owner_public: X25519PublicKey,
}

impl ModelOwnerDistributor {
    /// Creates a new ModelOwnerDistributor with fresh key material.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let (owner_secret, owner_public) = generate_x25519_keypair(&mut rng);
        Self {
            distributor: ShareDistributor::new(),
            owner_secret,
            owner_public,
        }
    }

    /// Creates with a deterministic seed (for testing).
    pub fn with_seed(seed: u64) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let (owner_secret, owner_public) = generate_x25519_keypair(&mut rng);
        Self {
            distributor: ShareDistributor::with_seed(seed.wrapping_add(1)),
            owner_secret,
            owner_public,
        }
    }

    /// Returns the owner's public key for workers to encrypt shares to.
    pub fn public_key(&self) -> X25519PublicKey {
        self.owner_public
    }

    /// Distributes model weights to workers.
    ///
    /// Splits the weight vector into additive shares, encrypts each to the
    /// corresponding worker's public key, and computes the initial Pedersen
    /// commitment.
    pub fn distribute(
        &mut self,
        weights: &[f64],
        shape: &[usize],
        worker_public_keys: &[X25519PublicKey],
        worker_ids: &[PartyId],
    ) -> Result<DistributionResult, helix_mpc::MPCError> {
        self.distributor.distribute(weights, shape, worker_public_keys, worker_ids)
    }

    /// Reconstructs trained weights from encrypted worker shares.
    ///
    /// Optionally verifies the result against a checkpoint commitment.
    pub fn reconstruct(
        &self,
        encrypted_shares: &[EncryptedShare],
        checkpoint: Option<&VectorCommitment>,
        checkpoint_blindings: Option<&[Fr]>,
    ) -> Result<Vec<f64>, helix_mpc::MPCError> {
        let reconstructor = WeightReconstructor::new(self.owner_secret.clone());
        reconstructor.reconstruct(encrypted_shares, checkpoint, checkpoint_blindings)
    }
}

impl Default for ModelOwnerDistributor {
    fn default() -> Self {
        Self::new()
    }
}
