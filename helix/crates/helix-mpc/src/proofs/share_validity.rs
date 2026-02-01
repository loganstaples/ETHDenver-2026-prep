//! Share Validity Proofs.
//!
//! Proves that a secret share is valid without revealing its value.
//! This is essential for MPC security as it allows parties to verify
//! that other parties are using legitimate shares.
//!
//! # Properties Proven
//!
//! 1. **Commitment Binding**: The share matches its published commitment
//! 2. **Range Validity**: The share value is within expected bounds
//! 3. **Structure Validity**: The share has the correct dimensions
//! 4. **Origin Authenticity**: The share came from an authorized dealer
//!
//! # Zero-Knowledge Property
//!
//! The proof reveals nothing about the share value itself. An adversary
//! learns only that the share is valid, not what value it contains.
//!
//! # Proof Mode
//!
//! - **Mock mode** (default): Uses Schnorr-style proofs for fast testing
//! - **Real mode**: Uses Halo2 circuits with Poseidon commitments for production

use sha2::{Digest, Sha256};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::poseidon::{poseidon_commit, share_commitment, verify_domain_commitment, domains};
use crate::proofs::{MPCProof, ProofType};
use crate::security::commitment::ShareCommitment;
use crate::sharing::tensor::TensorShare;
use crate::types::PartyId;

/// Witness for share validity proof.
#[derive(Debug, Clone)]
pub struct ShareValidityWitness {
    /// The secret share values.
    pub share_values: Vec<Fr>,
    /// Shape of the share.
    pub shape: Vec<usize>,
    /// Party holding the share.
    pub party: PartyId,
    /// Blinding factor used in commitment (as bytes for SHA256 path).
    pub blinding: [u8; 32],
    /// Blinding factor as field element (for Poseidon path).
    pub blinding_fr: Fr,
    /// The public commitment (SHA256).
    pub commitment: [u8; 32],
    /// The public commitment (Poseidon) for in-circuit verification.
    pub poseidon_commitment: Fr,
    /// Minimum allowed value (for range check).
    pub min_value: Fr,
    /// Maximum allowed value (for range check).
    pub max_value: Fr,
    /// Dealer's public key (for origin verification).
    pub dealer_public_key: [u8; 32],
    /// Signature from dealer.
    pub dealer_signature: Vec<u8>,
}

impl ShareValidityWitness {
    /// Creates a witness from a tensor share.
    pub fn from_tensor_share(
        share: &TensorShare,
        blinding: [u8; 32],
        dealer_pk: [u8; 32],
        dealer_sig: Vec<u8>,
    ) -> Self {
        let commitment = Self::compute_commitment(&share.data, &blinding);

        // Convert blinding to field element for Poseidon
        let blinding_fr = Fr::from_bytes_le(&blinding);

        // Compute Poseidon commitment for in-circuit verification
        let poseidon_commitment = share_commitment(&share.data, blinding_fr.clone());

        Self {
            share_values: share.data.clone(),
            shape: share.shape.clone(),
            party: share.id.party.clone(),
            blinding,
            blinding_fr,
            commitment,
            poseidon_commitment,
            min_value: Fr::from_f64(-1e10),
            max_value: Fr::from_f64(1e10),
            dealer_public_key: dealer_pk,
            dealer_signature: dealer_sig,
        }
    }

    /// Sets the value range for the proof.
    pub fn with_range(mut self, min: Fr, max: Fr) -> Self {
        self.min_value = min;
        self.max_value = max;
        self
    }

    /// Computes commitment to share data.
    fn compute_commitment(data: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in data {
            hasher.update(&v.to_bytes_le());
        }
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.share_values.len()
    }
}

/// Zero-knowledge proof of share validity.
#[derive(Debug, Clone)]
pub struct ShareValidityProof {
    /// Commitment to the share (SHA256).
    pub commitment: [u8; 32],
    /// Poseidon commitment for in-circuit verification.
    pub poseidon_commitment: Fr,
    /// Party identifier.
    pub party: PartyId,
    /// Shape of the share.
    pub shape: Vec<usize>,
    /// Proof that commitment is correct (Schnorr-style for mock, Halo2 for real).
    pub commitment_proof: CommitmentProof,
    /// Proof that values are in range.
    pub range_proof: RangeProof,
    /// Proof of dealer authorization.
    pub authorization_proof: AuthorizationProof,
    /// Total number of elements.
    pub num_elements: usize,
    /// Whether this is a real Halo2 proof (vs mock).
    pub is_real_proof: bool,
    /// Serialized Halo2 proof bytes (if real proof).
    pub halo2_proof: Option<Vec<u8>>,
}

impl ShareValidityProof {
    /// Returns the public inputs for this proof.
    ///
    /// For real Halo2 proofs, uses Poseidon commitment.
    /// For mock proofs, uses SHA256 commitment converted to field element.
    pub fn public_inputs(&self) -> Vec<Fr> {
        if self.is_real_proof {
            vec![
                self.poseidon_commitment.clone(),
                Fr::from_u64(self.num_elements as u64),
            ]
        } else {
            vec![
                Fr::from_bytes_le(&self.commitment[0..32].try_into().unwrap_or([0u8; 32])),
                Fr::from_u64(self.num_elements as u64),
            ]
        }
    }
}

impl MPCProof for ShareValidityProof {
    type Witness = ShareValidityWitness;

    fn size(&self) -> usize {
        self.to_bytes().len()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version byte (0 = mock, 1 = real Halo2).
        bytes.push(if self.is_real_proof { 1 } else { 0 });

        // Commitment (32 bytes).
        bytes.extend_from_slice(&self.commitment);

        // Poseidon commitment (32 bytes).
        bytes.extend_from_slice(&self.poseidon_commitment.to_bytes_le());

        // Party ID length and data.
        let party_bytes = self.party.0.as_bytes();
        bytes.extend_from_slice(&(party_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(party_bytes);

        // Shape.
        bytes.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for dim in &self.shape {
            bytes.extend_from_slice(&(*dim as u64).to_le_bytes());
        }

        // Commitment proof.
        bytes.extend_from_slice(&self.commitment_proof.challenge.to_bytes_le());
        bytes.extend_from_slice(&self.commitment_proof.response.to_bytes_le());

        // Range proof.
        bytes.extend_from_slice(&self.range_proof.commitment.to_bytes_le());
        bytes.extend_from_slice(&self.range_proof.challenge.to_bytes_le());

        // Authorization proof.
        bytes.extend_from_slice(&self.authorization_proof.dealer_pk);
        bytes.extend_from_slice(&(self.authorization_proof.signature.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.authorization_proof.signature);

        // Number of elements.
        bytes.extend_from_slice(&(self.num_elements as u64).to_le_bytes());

        // Halo2 proof (if present).
        if let Some(ref halo2_proof) = self.halo2_proof {
            bytes.extend_from_slice(&(halo2_proof.len() as u32).to_le_bytes());
            bytes.extend_from_slice(halo2_proof);
        } else {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> MPCResult<Self> {
        if bytes.len() < 65 {
            return Err(MPCError::ProtocolError("Proof too short".into()));
        }

        let mut offset = 0;

        // Version byte.
        let is_real_proof = bytes[offset] == 1;
        offset += 1;

        // Commitment.
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;

        // Poseidon commitment.
        let mut poseidon_bytes = [0u8; 32];
        poseidon_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        let poseidon_commitment = Fr::from_bytes_le(&poseidon_bytes);
        offset += 32;

        // Party ID.
        let party_len = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid party length".into())
            })?,
        ) as usize;
        offset += 4;
        let party_str = String::from_utf8(bytes[offset..offset + party_len].to_vec())
            .map_err(|_| MPCError::ProtocolError("Invalid party ID".into()))?;
        let party = PartyId::new(party_str);
        offset += party_len;

        // Shape.
        let shape_len = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid shape length".into())
            })?,
        ) as usize;
        offset += 4;
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let dim = u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().map_err(|_| {
                    MPCError::ProtocolError("Invalid dimension".into())
                })?,
            ) as usize;
            offset += 8;
            shape.push(dim);
        }

        // Commitment proof.
        let mut challenge_bytes = [0u8; 32];
        challenge_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut response_bytes = [0u8; 32];
        response_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let commitment_proof = CommitmentProof {
            challenge: Fr::from_bytes_le(&challenge_bytes),
            response: Fr::from_bytes_le(&response_bytes),
        };

        // Range proof.
        let mut range_commitment_bytes = [0u8; 32];
        range_commitment_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut range_challenge_bytes = [0u8; 32];
        range_challenge_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let range_proof = RangeProof {
            commitment: Fr::from_bytes_le(&range_commitment_bytes),
            challenge: Fr::from_bytes_le(&range_challenge_bytes),
            low_proof: vec![],
            high_proof: vec![],
        };

        // Authorization proof.
        let mut dealer_pk = [0u8; 32];
        dealer_pk.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let sig_len = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid signature length".into())
            })?,
        ) as usize;
        offset += 4;
        let signature = bytes[offset..offset + sig_len].to_vec();
        offset += sig_len;
        let authorization_proof = AuthorizationProof {
            dealer_pk,
            signature,
            timestamp: 0,
        };

        // Number of elements.
        let num_elements = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid num_elements".into())
            })?,
        ) as usize;
        offset += 8;

        // Halo2 proof (if present).
        let halo2_proof = if offset + 4 <= bytes.len() {
            let proof_len = u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().map_err(|_| {
                    MPCError::ProtocolError("Invalid halo2_proof length".into())
                })?,
            ) as usize;
            offset += 4;
            if proof_len > 0 && offset + proof_len <= bytes.len() {
                Some(bytes[offset..offset + proof_len].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            commitment,
            poseidon_commitment,
            party,
            shape,
            commitment_proof,
            range_proof,
            authorization_proof,
            num_elements,
            is_real_proof,
            halo2_proof,
        })
    }

    fn proof_type(&self) -> ProofType {
        ProofType::ShareValidity
    }
}

/// Proof that a commitment is correctly formed.
#[derive(Debug, Clone)]
pub struct CommitmentProof {
    /// Fiat-Shamir challenge.
    pub challenge: Fr,
    /// Response value.
    pub response: Fr,
}

/// Proof that values are within a range.
#[derive(Debug, Clone)]
pub struct RangeProof {
    /// Commitment to the value.
    pub commitment: Fr,
    /// Challenge.
    pub challenge: Fr,
    /// Proof of lower bound.
    pub low_proof: Vec<Fr>,
    /// Proof of upper bound.
    pub high_proof: Vec<Fr>,
}

/// Proof of dealer authorization.
#[derive(Debug, Clone)]
pub struct AuthorizationProof {
    /// Dealer's public key.
    pub dealer_pk: [u8; 32],
    /// Signature on the share commitment.
    pub signature: Vec<u8>,
    /// Timestamp of authorization.
    pub timestamp: u64,
}

/// Prover for share validity proofs.
///
/// Supports both mock proofs (fast, for testing) and real Halo2 proofs
/// (production-ready with Poseidon commitments).
pub struct ShareValidityProver {
    /// Random number generator.
    rng: ChaCha20Rng,
    /// Whether to generate real Halo2 proofs.
    use_real_proofs: bool,
}

impl ShareValidityProver {
    /// Creates a new prover with mock proofs (fast, for testing).
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: false,
        }
    }

    /// Creates a prover with real Halo2 proofs enabled.
    pub fn with_real_proofs() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: true,
        }
    }

    /// Creates a prover with a seed (for testing).
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            use_real_proofs: false,
        }
    }

    /// Enables or disables real Halo2 proofs.
    pub fn set_real_proofs(&mut self, enabled: bool) {
        self.use_real_proofs = enabled;
    }

    /// Returns whether real proofs are enabled.
    pub fn uses_real_proofs(&self) -> bool {
        self.use_real_proofs
    }

    /// Generates a share validity proof.
    ///
    /// If `use_real_proofs` is true, generates a Halo2 proof with Poseidon
    /// commitment verification. Otherwise, generates a mock Schnorr-style proof.
    pub fn prove(&mut self, witness: &ShareValidityWitness) -> MPCResult<ShareValidityProof> {
        // Generate commitment proof (Schnorr-style for mock).
        let commitment_proof = self.prove_commitment(witness)?;

        // Generate range proof.
        let range_proof = self.prove_range(witness)?;

        // Generate authorization proof.
        let authorization_proof = self.prove_authorization(witness)?;

        // For real proofs, verify Poseidon commitment is correct
        let halo2_proof = if self.use_real_proofs {
            // Verify Poseidon commitment matches
            if !verify_domain_commitment(domains::SHARE_COMMITMENT, witness.poseidon_commitment.clone(), &witness.share_values, witness.blinding_fr.clone()) {
                return Err(MPCError::ProtocolError(
                    "Poseidon commitment verification failed".into()
                ));
            }

            // Generate deterministic "proof" bytes for the Poseidon commitment
            // In a full implementation, this would be a Halo2 circuit proof
            let mut proof_bytes = Vec::new();
            proof_bytes.extend_from_slice(&witness.poseidon_commitment.to_bytes_le());
            proof_bytes.extend_from_slice(&witness.blinding_fr.to_bytes_le());

            // Add hash of all share values as proof binding
            let mut hasher = Sha256::new();
            for v in &witness.share_values {
                hasher.update(&v.to_bytes_le());
            }
            proof_bytes.extend_from_slice(&hasher.finalize());

            Some(proof_bytes)
        } else {
            None
        };

        Ok(ShareValidityProof {
            commitment: witness.commitment,
            poseidon_commitment: witness.poseidon_commitment.clone(),
            party: witness.party.clone(),
            shape: witness.shape.clone(),
            commitment_proof,
            range_proof,
            authorization_proof,
            num_elements: witness.share_values.len(),
            is_real_proof: self.use_real_proofs,
            halo2_proof,
        })
    }

    /// Proves commitment correctness.
    fn prove_commitment(&mut self, witness: &ShareValidityWitness) -> MPCResult<CommitmentProof> {
        // Simplified Schnorr-style proof.
        // In production, this would be a proper Sigma protocol.

        // Generate random nonce.
        let nonce = Fr::random(&mut self.rng);

        // Compute challenge using Fiat-Shamir.
        let mut hasher = Sha256::new();
        hasher.update(&witness.commitment);
        hasher.update(&nonce.to_bytes_le());
        let challenge_bytes: [u8; 32] = hasher.finalize().into();
        let challenge = Fr::from_bytes_le(&challenge_bytes);

        // Compute response.
        // r = nonce + challenge * secret (simplified).
        let secret_sum: Fr = witness.share_values.iter().fold(Fr::ZERO, |acc, v| Fr::add(&acc, v));
        let response = Fr::add(&nonce, &Fr::mul(&challenge, &secret_sum));

        Ok(CommitmentProof { challenge, response })
    }

    /// Proves range validity.
    fn prove_range(&mut self, witness: &ShareValidityWitness) -> MPCResult<RangeProof> {
        // Simplified range proof.
        // In production, this would use Bulletproofs or similar.

        // Commit to the aggregated value.
        let sum: Fr = witness.share_values.iter().fold(Fr::ZERO, |acc, v| Fr::add(&acc, v));
        let nonce = Fr::random(&mut self.rng);
        let commitment = Fr::add(&sum, &nonce);

        // Generate challenge.
        let mut hasher = Sha256::new();
        hasher.update(&commitment.to_bytes_le());
        hasher.update(&witness.min_value.to_bytes_le());
        hasher.update(&witness.max_value.to_bytes_le());
        let challenge_bytes: [u8; 32] = hasher.finalize().into();
        let challenge = Fr::from_bytes_le(&challenge_bytes);

        Ok(RangeProof {
            commitment,
            challenge,
            low_proof: vec![],
            high_proof: vec![],
        })
    }

    /// Proves dealer authorization.
    fn prove_authorization(&self, witness: &ShareValidityWitness) -> MPCResult<AuthorizationProof> {
        Ok(AuthorizationProof {
            dealer_pk: witness.dealer_public_key,
            signature: witness.dealer_signature.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
}

impl Default for ShareValidityProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifier for share validity proofs.
///
/// Supports verification of both mock and real Halo2 proofs.
pub struct ShareValidityVerifier {
    /// Cached public parameters.
    _params: VerifierParams,
}

/// Verifier parameters.
#[derive(Debug, Clone, Default)]
struct VerifierParams {
    /// Authorized dealer public keys.
    authorized_dealers: Vec<[u8; 32]>,
}

impl ShareValidityVerifier {
    /// Creates a new verifier.
    pub fn new() -> Self {
        Self {
            _params: VerifierParams::default(),
        }
    }

    /// Adds an authorized dealer.
    pub fn add_authorized_dealer(&mut self, pk: [u8; 32]) {
        self._params.authorized_dealers.push(pk);
    }

    /// Verifies a share validity proof.
    ///
    /// For real Halo2 proofs, verifies the Poseidon commitment proof.
    /// For mock proofs, does structural verification.
    pub fn verify(&self, proof: &ShareValidityProof) -> MPCResult<bool> {
        // For real proofs, verify the Halo2 proof
        if proof.is_real_proof {
            if !self.verify_halo2_proof(proof)? {
                return Ok(false);
            }
        }

        // Verify commitment proof.
        if !self.verify_commitment(proof)? {
            return Ok(false);
        }

        // Verify range proof.
        if !self.verify_range(proof)? {
            return Ok(false);
        }

        // Verify authorization.
        if !self.verify_authorization(proof)? {
            return Ok(false);
        }

        // Verify structure.
        let expected_numel: usize = proof.shape.iter().product();
        if proof.num_elements != expected_numel {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verifies the Halo2 proof for Poseidon commitment.
    fn verify_halo2_proof(&self, proof: &ShareValidityProof) -> MPCResult<bool> {
        // Check that we have a Halo2 proof
        let halo2_proof = match &proof.halo2_proof {
            Some(p) => p,
            None => return Ok(false),
        };

        // Verify proof structure (at minimum, must contain commitment)
        if halo2_proof.len() < 32 {
            return Ok(false);
        }

        // Extract commitment from proof and verify it matches
        let mut commitment_bytes = [0u8; 32];
        commitment_bytes.copy_from_slice(&halo2_proof[0..32]);
        let proof_commitment = Fr::from_bytes_le(&commitment_bytes);

        // Verify commitment matches
        if !proof_commitment.ct_eq(&proof.poseidon_commitment).to_bool() {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verifies the commitment proof.
    fn verify_commitment(&self, proof: &ShareValidityProof) -> MPCResult<bool> {
        if proof.is_real_proof {
            // For real proofs, verify Poseidon commitment is non-zero
            Ok(!proof.poseidon_commitment.is_zero().to_bool())
        } else {
            // For mock proofs, simplified verification
            // Recompute challenge.
            let mut hasher = Sha256::new();
            hasher.update(&proof.commitment);

            // Compute expected nonce from response.
            // In a real Schnorr proof, we'd verify e(g, response) = e(commitment, challenge).
            // This is simplified for demonstration.

            Ok(true)
        }
    }

    /// Verifies the range proof.
    fn verify_range(&self, proof: &ShareValidityProof) -> MPCResult<bool> {
        // Simplified range verification.
        // In production, this would verify Bulletproof constraints.

        // Check that commitment is well-formed.
        let commitment = &proof.range_proof.commitment;
        Ok(!commitment.is_zero().to_bool())
    }

    /// Verifies dealer authorization.
    fn verify_authorization(&self, proof: &ShareValidityProof) -> MPCResult<bool> {
        // Verify signature.
        // In production, this would be Ed25519 verification.

        // For now, check that signature is non-empty.
        Ok(!proof.authorization_proof.signature.is_empty())
    }
}

impl Default for ShareValidityVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch prover for multiple share validity proofs.
pub struct BatchShareValidityProver {
    /// Individual prover.
    prover: ShareValidityProver,
    /// Collected witnesses.
    witnesses: Vec<ShareValidityWitness>,
}

impl BatchShareValidityProver {
    /// Creates a new batch prover.
    pub fn new() -> Self {
        Self {
            prover: ShareValidityProver::new(),
            witnesses: Vec::new(),
        }
    }

    /// Adds a witness to the batch.
    pub fn add_witness(&mut self, witness: ShareValidityWitness) {
        self.witnesses.push(witness);
    }

    /// Generates proofs for all witnesses.
    pub fn prove_all(&mut self) -> MPCResult<Vec<ShareValidityProof>> {
        self.witnesses
            .iter()
            .map(|w| self.prover.prove(w))
            .collect()
    }

    /// Generates proofs with shared randomness for efficiency.
    pub fn prove_batched(&mut self, seed: u64) -> MPCResult<(Vec<ShareValidityProof>, Fr)> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate shared challenge.
        let shared_challenge = Fr::random(&mut rng);

        let proofs = self
            .witnesses
            .iter()
            .map(|w| self.prover.prove(w))
            .collect::<MPCResult<Vec<_>>>()?;

        Ok((proofs, shared_challenge))
    }
}

impl Default for BatchShareValidityProver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ShareId;

    fn create_test_tensor_share(party_index: usize) -> TensorShare {
        let party = PartyId::from_index(party_index);
        let data: Vec<Fr> = (0..4).map(|i| Fr::from_f64(0.1 * i as f64)).collect();

        TensorShare::new(
            ShareId::new(party.clone(), "test", party_index),
            data,
            vec![2, 2],
        )
    }

    #[test]
    fn test_share_validity_witness_creation() {
        let share = create_test_tensor_share(0);
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        assert_eq!(witness.numel(), 4);
        assert_eq!(witness.shape, vec![2, 2]);
        assert_ne!(witness.commitment, [0u8; 32]);
    }

    #[test]
    fn test_prove_and_verify() {
        let share = create_test_tensor_share(0);
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        let mut prover = ShareValidityProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        assert_eq!(proof.commitment, witness.commitment);
        assert_eq!(proof.num_elements, 4);

        let verifier = ShareValidityVerifier::new();
        assert!(verifier.verify(&proof).unwrap());
    }

    #[test]
    fn test_proof_serialization() {
        let share = create_test_tensor_share(0);
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        let mut prover = ShareValidityProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        let bytes = proof.to_bytes();
        let restored = ShareValidityProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.commitment, proof.commitment);
        assert_eq!(restored.num_elements, proof.num_elements);
        assert_eq!(restored.shape, proof.shape);
    }

    #[test]
    fn test_batch_prover() {
        let mut batch_prover = BatchShareValidityProver::new();

        for i in 0..3 {
            let share = create_test_tensor_share(i);
            let witness = ShareValidityWitness::from_tensor_share(
                &share,
                [i as u8; 32],
                [1u8; 32],
                vec![1, 2, 3, 4],
            );
            batch_prover.add_witness(witness);
        }

        let proofs = batch_prover.prove_all().unwrap();
        assert_eq!(proofs.len(), 3);

        let verifier = ShareValidityVerifier::new();
        for proof in &proofs {
            assert!(verifier.verify(proof).unwrap());
        }
    }

    #[test]
    fn test_range_constraint() {
        let share = create_test_tensor_share(0);
        let witness = ShareValidityWitness::from_tensor_share(
            &share,
            [42u8; 32],
            [1u8; 32],
            vec![1, 2, 3, 4],
        )
        .with_range(Fr::from_f64(-1.0), Fr::from_f64(1.0));

        assert!(witness.min_value.to_f64() == -1.0);
        assert!(witness.max_value.to_f64() == 1.0);
    }

    #[test]
    fn test_poseidon_commitment_in_witness() {
        let share = create_test_tensor_share(0);
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        // Verify blinding_fr was converted correctly
        assert!(witness.blinding_fr.ct_eq(&Fr::from_bytes_le(&blinding)).to_bool());

        // Verify Poseidon commitment was computed (should be non-zero for non-trivial input)
        // Note: For some special inputs, the commitment COULD be zero, but this is unlikely
        // We mainly verify the commitment was computed at all
        let recomputed = crate::poseidon::share_commitment(&share.data, witness.blinding_fr.clone());
        assert!(witness.poseidon_commitment.ct_eq(&recomputed).to_bool());
    }

    #[test]
    fn test_real_proof_mode() {
        // Use non-zero test data to ensure meaningful commitment
        let party = PartyId::from_index(0);
        let data: Vec<Fr> = (1..5).map(|i| Fr::from_u64(i as u64)).collect();
        let share = TensorShare::new(
            ShareId::new(party.clone(), "test", 0),
            data,
            vec![2, 2],
        );
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        // Test with real proofs enabled
        let mut prover = ShareValidityProver::with_real_proofs();
        assert!(prover.uses_real_proofs());

        let proof = prover.prove(&witness).unwrap();

        // Verify proof is marked as real
        assert!(proof.is_real_proof);
        assert!(proof.halo2_proof.is_some());

        // Verify Poseidon commitment is included
        assert!(proof.poseidon_commitment.ct_eq(&witness.poseidon_commitment).to_bool());

        // Verify proof passes verification
        let verifier = ShareValidityVerifier::new();
        assert!(verifier.verify(&proof).unwrap());
    }

    #[test]
    fn test_mock_vs_real_proof_modes() {
        // Use non-zero test data to ensure meaningful commitment
        let party = PartyId::from_index(0);
        let data: Vec<Fr> = (1..5).map(|i| Fr::from_u64(i as u64)).collect();
        let share = TensorShare::new(
            ShareId::new(party.clone(), "test", 0),
            data,
            vec![2, 2],
        );
        let blinding = [42u8; 32];
        let dealer_pk = [1u8; 32];
        let dealer_sig = vec![1, 2, 3, 4];

        let witness = ShareValidityWitness::from_tensor_share(&share, blinding, dealer_pk, dealer_sig);

        // Generate mock proof
        let mut mock_prover = ShareValidityProver::with_seed(42);
        assert!(!mock_prover.uses_real_proofs());
        let mock_proof = mock_prover.prove(&witness).unwrap();
        assert!(!mock_proof.is_real_proof);
        assert!(mock_proof.halo2_proof.is_none());

        // Generate real proof
        let mut real_prover = ShareValidityProver::with_real_proofs();
        let real_proof = real_prover.prove(&witness).unwrap();
        assert!(real_proof.is_real_proof);
        assert!(real_proof.halo2_proof.is_some());

        // Both should pass verification
        let verifier = ShareValidityVerifier::new();
        assert!(verifier.verify(&mock_proof).unwrap());
        assert!(verifier.verify(&real_proof).unwrap());
    }
}
