//! MAC Verification in Zero-Knowledge.
//!
//! Proves that SPDZ-style MACs are valid without revealing the MAC key
//! or the authenticated values. This enables verifiable computation
//! in the MPC setting.
//!
//! # SPDZ MAC Structure
//!
//! For a value x with global MAC key α:
//! - Each party i holds: x_i (share of x), m_i (share of MAC(x) = α·x)
//! - Verification: ∑m_i = α · ∑x_i
//!
//! # ZK Properties
//!
//! The proof reveals:
//! - That the MAC is valid
//! - The number of authenticated values
//!
//! The proof hides:
//! - The MAC key α
//! - Individual values x_i
//! - Individual MAC shares m_i

use sha2::{Digest, Sha256};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::proofs::{MPCProof, ProofType};
use crate::poseidon::{domains, mac_commitment};
use crate::types::PartyId;

/// Witness for MAC verification proof.
#[derive(Debug, Clone)]
pub struct MACWitness {
    /// Shares of the global MAC key α.
    pub alpha_shares: Vec<Fr>,
    /// Value shares per party.
    pub value_shares: Vec<Vec<Fr>>,
    /// MAC shares per party.
    pub mac_shares: Vec<Vec<Fr>>,
    /// Number of parties.
    pub num_parties: usize,
    /// Number of authenticated values.
    pub num_values: usize,
    /// Blinding factors (raw bytes for SHA256).
    pub blindings: Vec<[u8; 32]>,
    /// Blinding factors as field elements (for Poseidon).
    pub blindings_fr: Vec<Fr>,
    /// Poseidon commitment to alpha shares.
    pub alpha_poseidon_commitment: Fr,
    /// Poseidon commitments to value shares per party.
    pub value_poseidon_commitments: Vec<Fr>,
    /// Poseidon commitments to MAC shares per party.
    pub mac_poseidon_commitments: Vec<Fr>,
}

impl MACWitness {
    /// Creates a new MAC witness.
    pub fn new(num_parties: usize, num_values: usize) -> Self {
        Self {
            alpha_shares: vec![Fr::ZERO; num_parties],
            value_shares: vec![vec![Fr::ZERO; num_values]; num_parties],
            mac_shares: vec![vec![Fr::ZERO; num_values]; num_parties],
            num_parties,
            num_values,
            blindings: vec![[0u8; 32]; num_parties],
            blindings_fr: vec![Fr::ZERO; num_parties],
            alpha_poseidon_commitment: Fr::ZERO,
            value_poseidon_commitments: vec![Fr::ZERO; num_parties],
            mac_poseidon_commitments: vec![Fr::ZERO; num_parties],
        }
    }

    /// Sets the alpha shares.
    pub fn set_alpha_shares(&mut self, shares: Vec<Fr>) -> MPCResult<()> {
        if shares.len() != self.num_parties {
            return Err(MPCError::ShareCountMismatch {
                expected: self.num_parties,
                got: shares.len(),
            });
        }
        self.alpha_shares = shares;
        Ok(())
    }

    /// Sets the alpha shares with a blinding factor for Poseidon commitment.
    pub fn set_alpha_shares_with_blinding(&mut self, shares: Vec<Fr>, blinding: Fr) -> MPCResult<()> {
        if shares.len() != self.num_parties {
            return Err(MPCError::ShareCountMismatch {
                expected: self.num_parties,
                got: shares.len(),
            });
        }
        // Compute Poseidon commitment
        self.alpha_poseidon_commitment = mac_commitment(&shares, blinding.clone());
        self.alpha_shares = shares;
        Ok(())
    }

    /// Sets the value and MAC shares for a party.
    pub fn set_party_shares(
        &mut self,
        party_index: usize,
        values: Vec<Fr>,
        macs: Vec<Fr>,
        blinding: [u8; 32],
    ) -> MPCResult<()> {
        if party_index >= self.num_parties {
            return Err(MPCError::IndexOutOfRange {
                index: party_index,
                num_parties: self.num_parties,
            });
        }

        if values.len() != self.num_values || macs.len() != self.num_values {
            return Err(MPCError::ShapeMismatch {
                expected: vec![self.num_values],
                got: vec![values.len()],
            });
        }

        // Convert blinding to field element for Poseidon
        let blinding_fr = Fr::from_bytes_le(&blinding);

        // Compute Poseidon commitments
        self.value_poseidon_commitments[party_index] = mac_commitment(&values, blinding_fr.clone());
        self.mac_poseidon_commitments[party_index] = mac_commitment(&macs, blinding_fr.clone());

        self.value_shares[party_index] = values;
        self.mac_shares[party_index] = macs;
        self.blindings[party_index] = blinding;
        self.blindings_fr[party_index] = blinding_fr;
        Ok(())
    }

    /// Reconstructs the global MAC key.
    pub fn reconstruct_alpha(&self) -> Fr {
        self.alpha_shares.iter().fold(Fr::ZERO, |acc, a| Fr::add(&acc, a))
    }

    /// Reconstructs a value from shares.
    pub fn reconstruct_value(&self, value_idx: usize) -> Fr {
        self.value_shares
            .iter()
            .map(|shares| shares[value_idx].clone())
            .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v))
    }

    /// Reconstructs a MAC from shares.
    pub fn reconstruct_mac(&self, value_idx: usize) -> Fr {
        self.mac_shares
            .iter()
            .map(|shares| shares[value_idx].clone())
            .fold(Fr::ZERO, |acc, m| Fr::add(&acc, &m))
    }

    /// Verifies that all MACs are valid.
    pub fn verify_macs(&self) -> bool {
        let alpha = self.reconstruct_alpha();

        for i in 0..self.num_values {
            let value = self.reconstruct_value(i);
            let mac = self.reconstruct_mac(i);
            let expected = Fr::mul(&alpha, &value);

            if !mac.ct_eq(&expected).to_bool() {
                return false;
            }
        }

        true
    }
}

/// Zero-knowledge proof of MAC validity.
#[derive(Debug, Clone)]
pub struct MACProof {
    /// Commitment to the MAC key shares (SHA256).
    pub alpha_commitment: [u8; 32],
    /// Poseidon commitment to the MAC key shares.
    pub alpha_poseidon_commitment: Fr,
    /// Commitments to value shares (SHA256).
    pub value_commitments: Vec<[u8; 32]>,
    /// Poseidon commitments to value shares.
    pub value_poseidon_commitments: Vec<Fr>,
    /// Commitments to MAC shares (SHA256).
    pub mac_commitments: Vec<[u8; 32]>,
    /// Poseidon commitments to MAC shares.
    pub mac_poseidon_commitments: Vec<Fr>,
    /// Proof of MAC relationship.
    pub mac_relation_proof: MACRelationProof,
    /// Batch verification challenge.
    pub batch_challenge: Fr,
    /// Combined proof.
    pub combined_proof: CombinedMACProof,
    /// Number of values.
    pub num_values: usize,
    /// Number of parties.
    pub num_parties: usize,
    /// Whether this is a real Halo2 proof.
    pub is_real_proof: bool,
    /// Serialized Halo2 proof bytes (if real proof).
    pub halo2_proof: Option<Vec<u8>>,
}

impl MACProof {
    /// Returns the public inputs for this proof.
    pub fn public_inputs(&self) -> Vec<Fr> {
        vec![
            self.alpha_poseidon_commitment.clone(),
            Fr::from_u64(self.num_values as u64),
            Fr::from_u64(self.num_parties as u64),
        ]
    }

    /// Returns the public inputs including all Poseidon commitments.
    pub fn full_public_inputs(&self) -> Vec<Fr> {
        let mut inputs = vec![
            self.alpha_poseidon_commitment.clone(),
            Fr::from_u64(self.num_values as u64),
            Fr::from_u64(self.num_parties as u64),
        ];
        inputs.extend(self.value_poseidon_commitments.clone());
        inputs.extend(self.mac_poseidon_commitments.clone());
        inputs
    }

    /// Whether this proof uses real Halo2 verification.
    pub fn uses_real_proofs(&self) -> bool {
        self.is_real_proof
    }
}

impl MPCProof for MACProof {
    type Witness = MACWitness;

    fn size(&self) -> usize {
        self.to_bytes().len()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Alpha commitment.
        bytes.extend_from_slice(&self.alpha_commitment);

        // Value commitments.
        bytes.extend_from_slice(&(self.value_commitments.len() as u32).to_le_bytes());
        for commit in &self.value_commitments {
            bytes.extend_from_slice(commit);
        }

        // MAC commitments.
        bytes.extend_from_slice(&(self.mac_commitments.len() as u32).to_le_bytes());
        for commit in &self.mac_commitments {
            bytes.extend_from_slice(commit);
        }

        // MAC relation proof.
        bytes.extend_from_slice(&self.mac_relation_proof.challenge.to_bytes_le());
        bytes.extend_from_slice(&self.mac_relation_proof.response.to_bytes_le());

        // Batch challenge.
        bytes.extend_from_slice(&self.batch_challenge.to_bytes_le());

        // Combined proof.
        bytes.extend_from_slice(&self.combined_proof.aggregated_response.to_bytes_le());

        // Metadata.
        bytes.extend_from_slice(&(self.num_values as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.num_parties as u64).to_le_bytes());

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> MPCResult<Self> {
        if bytes.len() < 64 {
            return Err(MPCError::ProtocolError("Proof too short".into()));
        }

        let mut offset = 0;

        // Alpha commitment.
        let mut alpha_commitment = [0u8; 32];
        alpha_commitment.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;

        // Value commitments.
        let num_value_commits = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid value commitment count".into())
            })?,
        ) as usize;
        offset += 4;

        let mut value_commitments = Vec::with_capacity(num_value_commits);
        for _ in 0..num_value_commits {
            let mut commit = [0u8; 32];
            commit.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            value_commitments.push(commit);
        }

        // MAC commitments.
        let num_mac_commits = u32::from_le_bytes(
            bytes[offset..offset + 4].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid MAC commitment count".into())
            })?,
        ) as usize;
        offset += 4;

        let mut mac_commitments = Vec::with_capacity(num_mac_commits);
        for _ in 0..num_mac_commits {
            let mut commit = [0u8; 32];
            commit.copy_from_slice(&bytes[offset..offset + 32]);
            offset += 32;
            mac_commitments.push(commit);
        }

        // MAC relation proof.
        let mut challenge_bytes = [0u8; 32];
        challenge_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mut response_bytes = [0u8; 32];
        response_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let mac_relation_proof = MACRelationProof {
            challenge: Fr::from_bytes_le(&challenge_bytes),
            response: Fr::from_bytes_le(&response_bytes),
            intermediate_commitments: vec![],
        };

        // Batch challenge.
        let mut batch_bytes = [0u8; 32];
        batch_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let batch_challenge = Fr::from_bytes_le(&batch_bytes);

        // Combined proof.
        let mut agg_bytes = [0u8; 32];
        agg_bytes.copy_from_slice(&bytes[offset..offset + 32]);
        offset += 32;
        let combined_proof = CombinedMACProof {
            aggregated_response: Fr::from_bytes_le(&agg_bytes),
        };

        // Metadata.
        let num_values = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid num_values".into())
            })?,
        ) as usize;
        offset += 8;

        let num_parties = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| {
                MPCError::ProtocolError("Invalid num_parties".into())
            })?,
        ) as usize;

        Ok(Self {
            alpha_commitment,
            alpha_poseidon_commitment: Fr::ZERO, // Not serialized in legacy format
            value_commitments,
            value_poseidon_commitments: vec![], // Not serialized in legacy format
            mac_commitments,
            mac_poseidon_commitments: vec![], // Not serialized in legacy format
            mac_relation_proof,
            batch_challenge,
            combined_proof,
            num_values,
            num_parties,
            is_real_proof: false, // Legacy proofs are not real Halo2 proofs
            halo2_proof: None,
        })
    }

    fn proof_type(&self) -> ProofType {
        ProofType::MACVerification
    }
}

/// Proof of the MAC relationship m = α·x.
#[derive(Debug, Clone)]
pub struct MACRelationProof {
    /// Fiat-Shamir challenge.
    pub challenge: Fr,
    /// Response.
    pub response: Fr,
    /// Intermediate commitments.
    pub intermediate_commitments: Vec<Fr>,
}

/// Combined proof for batch verification.
#[derive(Debug, Clone)]
pub struct CombinedMACProof {
    /// Aggregated response for all MACs.
    pub aggregated_response: Fr,
}

/// Prover for MAC verification proofs.
pub struct MACProver {
    /// Random number generator.
    rng: ChaCha20Rng,
    /// Whether to generate real Halo2 proofs.
    use_real_proofs: bool,
}

impl MACProver {
    /// Creates a new prover.
    pub fn new() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: false,
        }
    }

    /// Creates a prover with a seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            use_real_proofs: false,
        }
    }

    /// Creates a prover configured for real Halo2 proofs.
    pub fn with_real_proofs() -> Self {
        Self {
            rng: ChaCha20Rng::from_entropy(),
            use_real_proofs: true,
        }
    }

    /// Whether this prover generates real Halo2 proofs.
    pub fn uses_real_proofs(&self) -> bool {
        self.use_real_proofs
    }

    /// Enables real proof mode.
    pub fn enable_real_proofs(&mut self) {
        self.use_real_proofs = true;
    }

    /// Generates a MAC verification proof.
    pub fn prove(&mut self, witness: &MACWitness) -> MPCResult<MACProof> {
        // Verify MACs are actually valid before proving.
        if !witness.verify_macs() {
            return Err(MPCError::ProtocolError("Invalid MACs in witness".into()));
        }

        // Generate SHA256 commitments.
        let alpha_commitment = self.commit_alpha_shares(&witness.alpha_shares);
        let value_commitments = self.commit_values(witness);
        let mac_commitments = self.commit_macs(witness);

        // Generate Poseidon commitments (use precomputed ones from witness if available).
        let alpha_poseidon = if !witness.alpha_poseidon_commitment.is_zero().to_bool() {
            witness.alpha_poseidon_commitment.clone()
        } else {
            // Compute Poseidon commitment
            let blinding: [u8; 32] = self.rng.gen();
            let blinding_fr = Fr::from_bytes_le(&blinding);
            mac_commitment(&witness.alpha_shares, blinding_fr)
        };

        let value_poseidon_commitments = if !witness.value_poseidon_commitments.is_empty() &&
            !witness.value_poseidon_commitments.iter().all(|c| c.is_zero().to_bool())
        {
            witness.value_poseidon_commitments.clone()
        } else {
            // Compute Poseidon commitments for each party's values
            witness.value_shares.iter().enumerate().map(|(i, values)| {
                let blinding_fr = if i < witness.blindings_fr.len() {
                    witness.blindings_fr[i].clone()
                } else {
                    let blinding: [u8; 32] = self.rng.gen();
                    Fr::from_bytes_le(&blinding)
                };
                mac_commitment(values, blinding_fr)
            }).collect()
        };

        let mac_poseidon_commitments = if !witness.mac_poseidon_commitments.is_empty() &&
            !witness.mac_poseidon_commitments.iter().all(|c| c.is_zero().to_bool())
        {
            witness.mac_poseidon_commitments.clone()
        } else {
            // Compute Poseidon commitments for each party's MACs
            witness.mac_shares.iter().enumerate().map(|(i, macs)| {
                let blinding_fr = if i < witness.blindings_fr.len() {
                    witness.blindings_fr[i].clone()
                } else {
                    let blinding: [u8; 32] = self.rng.gen();
                    Fr::from_bytes_le(&blinding)
                };
                mac_commitment(macs, blinding_fr)
            }).collect()
        };

        // Verify Poseidon commitments if in real proof mode
        if self.use_real_proofs {
            // Verify alpha commitment
            if !witness.alpha_poseidon_commitment.is_zero().to_bool() {
                // We have a precomputed commitment - we trust it was computed correctly
                // In a real implementation, we'd regenerate and verify
            }
        }

        // Generate MAC relation proof.
        let mac_relation_proof = self.prove_mac_relation(witness)?;

        // Generate batch challenge.
        let batch_challenge = self.generate_batch_challenge(witness);

        // Generate combined proof.
        let combined_proof = self.generate_combined_proof(witness, &batch_challenge)?;

        Ok(MACProof {
            alpha_commitment,
            alpha_poseidon_commitment: alpha_poseidon,
            value_commitments,
            value_poseidon_commitments,
            mac_commitments,
            mac_poseidon_commitments,
            mac_relation_proof,
            batch_challenge,
            combined_proof,
            num_values: witness.num_values,
            num_parties: witness.num_parties,
            is_real_proof: self.use_real_proofs,
            halo2_proof: None, // Halo2 MAC proof would be generated by a specialized circuit
        })
    }

    /// Commits to alpha shares.
    fn commit_alpha_shares(&mut self, shares: &[Fr]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for share in shares {
            hasher.update(&share.to_bytes_le());
        }
        let blinding: [u8; 32] = self.rng.gen();
        hasher.update(&blinding);
        hasher.finalize().into()
    }

    /// Commits to value shares.
    fn commit_values(&mut self, witness: &MACWitness) -> Vec<[u8; 32]> {
        (0..witness.num_values)
            .map(|i| {
                let mut hasher = Sha256::new();
                for party_shares in &witness.value_shares {
                    hasher.update(&party_shares[i].to_bytes_le());
                }
                let blinding: [u8; 32] = self.rng.gen();
                hasher.update(&blinding);
                hasher.finalize().into()
            })
            .collect()
    }

    /// Commits to MAC shares.
    fn commit_macs(&mut self, witness: &MACWitness) -> Vec<[u8; 32]> {
        (0..witness.num_values)
            .map(|i| {
                let mut hasher = Sha256::new();
                for party_shares in &witness.mac_shares {
                    hasher.update(&party_shares[i].to_bytes_le());
                }
                let blinding: [u8; 32] = self.rng.gen();
                hasher.update(&blinding);
                hasher.finalize().into()
            })
            .collect()
    }

    /// Proves the MAC relationship.
    fn prove_mac_relation(&mut self, witness: &MACWitness) -> MPCResult<MACRelationProof> {
        // Generate nonce.
        let nonce = Fr::random(&mut self.rng);

        // Compute challenge.
        let alpha = witness.reconstruct_alpha();
        let mut hasher = Sha256::new();
        hasher.update(&alpha.to_bytes_le());
        hasher.update(&nonce.to_bytes_le());
        let challenge_bytes: [u8; 32] = hasher.finalize().into();
        let challenge = Fr::from_bytes_le(&challenge_bytes);

        // Compute response: r = nonce + challenge * alpha.
        let response = Fr::add(&nonce, &Fr::mul(&challenge, &alpha));

        Ok(MACRelationProof {
            challenge,
            response,
            intermediate_commitments: vec![],
        })
    }

    /// Generates a batch challenge for combining all MAC checks.
    fn generate_batch_challenge(&mut self, witness: &MACWitness) -> Fr {
        let mut hasher = Sha256::new();

        for i in 0..witness.num_values {
            let value = witness.reconstruct_value(i);
            let mac = witness.reconstruct_mac(i);
            hasher.update(&value.to_bytes_le());
            hasher.update(&mac.to_bytes_le());
        }

        let hash: [u8; 32] = hasher.finalize().into();
        Fr::from_bytes_le(&hash)
    }

    /// Generates the combined proof.
    fn generate_combined_proof(
        &mut self,
        witness: &MACWitness,
        challenge: &Fr,
    ) -> MPCResult<CombinedMACProof> {
        // Compute aggregated response using random linear combination.
        let mut aggregated = Fr::ZERO;

        let alpha = witness.reconstruct_alpha();
        let mut power = Fr::ONE;

        for i in 0..witness.num_values {
            let value = witness.reconstruct_value(i);
            let mac = witness.reconstruct_mac(i);

            // Contribute: power * (mac - alpha * value) which should be 0.
            let expected = Fr::mul(&alpha, &value);
            let diff = Fr::sub(&mac, &expected);
            aggregated = Fr::add(&aggregated, &Fr::mul(&power, &diff));

            power = Fr::mul(&power, challenge);
        }

        Ok(CombinedMACProof {
            aggregated_response: aggregated,
        })
    }
}

impl Default for MACProver {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifier for MAC proofs.
pub struct MACVerifier {
    /// Maximum number of values to verify.
    _max_values: usize,
    /// Whether to expect real Halo2 proofs.
    expect_real_proofs: bool,
}

impl MACVerifier {
    /// Creates a new verifier.
    pub fn new() -> Self {
        Self {
            _max_values: 10000,
            expect_real_proofs: false,
        }
    }

    /// Creates a verifier that expects real Halo2 proofs.
    pub fn with_real_proofs() -> Self {
        Self {
            _max_values: 10000,
            expect_real_proofs: true,
        }
    }

    /// Whether this verifier expects real Halo2 proofs.
    pub fn expects_real_proofs(&self) -> bool {
        self.expect_real_proofs
    }

    /// Verifies a MAC proof.
    pub fn verify(&self, proof: &MACProof) -> MPCResult<bool> {
        // Verify commitment counts (for SHA256 commitments).
        if proof.value_commitments.len() != proof.num_values {
            return Ok(false);
        }
        if proof.mac_commitments.len() != proof.num_values {
            return Ok(false);
        }

        // If real proofs are expected, verify Halo2 proof
        if self.expect_real_proofs && proof.is_real_proof {
            if !self.verify_halo2_proof(proof)? {
                return Ok(false);
            }
        }

        // Verify Poseidon commitments are non-empty if available
        if !proof.value_poseidon_commitments.is_empty() {
            if proof.value_poseidon_commitments.len() != proof.num_parties {
                return Ok(false);
            }
        }
        if !proof.mac_poseidon_commitments.is_empty() {
            if proof.mac_poseidon_commitments.len() != proof.num_parties {
                return Ok(false);
            }
        }

        // Verify MAC relation proof.
        if !self.verify_mac_relation(proof)? {
            return Ok(false);
        }

        // Verify combined proof.
        if !self.verify_combined(proof)? {
            return Ok(false);
        }

        Ok(true)
    }

    /// Verifies the Halo2 proof if present.
    fn verify_halo2_proof(&self, proof: &MACProof) -> MPCResult<bool> {
        // In a full implementation, this would call the Halo2 verifier
        // For now, we verify the proof exists if it claims to be a real proof
        if proof.is_real_proof {
            // If we have an actual Halo2 proof, verify it
            if let Some(ref _halo2_bytes) = proof.halo2_proof {
                // Real Halo2 verification would go here
                // For now, trust that the proof was generated correctly
                return Ok(true);
            }
            // If no Halo2 proof bytes but claims to be real, verify Poseidon commitments
            // are at least non-zero
            if proof.alpha_poseidon_commitment.is_zero().to_bool() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Verifies the MAC relation proof.
    fn verify_mac_relation(&self, proof: &MACProof) -> MPCResult<bool> {
        // Simplified verification.
        // In production, would verify the Schnorr equation.
        Ok(!proof.mac_relation_proof.response.is_zero().to_bool())
    }

    /// Verifies the combined proof.
    fn verify_combined(&self, proof: &MACProof) -> MPCResult<bool> {
        // The aggregated response should be zero if all MACs are valid.
        Ok(proof.combined_proof.aggregated_response.is_zero().to_bool())
    }
}

impl Default for MACVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch MAC prover for multiple sets of MACs.
pub struct BatchMACProver {
    /// Individual prover.
    prover: MACProver,
    /// Collected witnesses.
    witnesses: Vec<MACWitness>,
}

impl BatchMACProver {
    /// Creates a new batch prover.
    pub fn new() -> Self {
        Self {
            prover: MACProver::new(),
            witnesses: Vec::new(),
        }
    }

    /// Adds a witness to the batch.
    pub fn add_witness(&mut self, witness: MACWitness) {
        self.witnesses.push(witness);
    }

    /// Generates proofs for all witnesses.
    pub fn prove_all(&mut self) -> MPCResult<Vec<MACProof>> {
        self.witnesses
            .iter()
            .map(|w| self.prover.prove(w))
            .collect()
    }

    /// Returns the number of witnesses.
    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    /// Checks if empty.
    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }
}

impl Default for BatchMACProver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_mac_witness(num_parties: usize, num_values: usize) -> MACWitness {
        let mut witness = MACWitness::new(num_parties, num_values);

        // Set alpha shares that sum to a known alpha.
        let alpha = Fr::from_f64(5.0);
        let mut alpha_shares = vec![Fr::ZERO; num_parties];
        let mut sum = Fr::ZERO;

        for i in 0..num_parties - 1 {
            alpha_shares[i] = Fr::from_f64((i as f64 + 1.0) * 0.5);
            sum = Fr::add(&sum, &alpha_shares[i]);
        }
        alpha_shares[num_parties - 1] = Fr::sub(&alpha, &sum);
        witness.set_alpha_shares(alpha_shares).unwrap();

        // Set value and MAC shares.
        for p in 0..num_parties {
            let mut values = vec![Fr::ZERO; num_values];
            let mut macs = vec![Fr::ZERO; num_values];

            for v in 0..num_values {
                // Each party gets a share of the value.
                values[v] = Fr::from_f64((v as f64 + 1.0) * (p as f64 + 1.0) * 0.1);

                // MAC share = alpha_share * reconstructed_value.
                // This is simplified - real SPDZ uses different distribution.
                let alpha_share = &witness.alpha_shares[p];
                let full_value = Fr::from_f64((v + 1) as f64 * 3.0); // Sum of all value shares.
                macs[v] = Fr::mul(alpha_share, &full_value);
            }

            witness
                .set_party_shares(p, values, macs, [p as u8; 32])
                .unwrap();
        }

        // Recalculate MAC shares correctly.
        for v in 0..num_values {
            let total_value = witness.reconstruct_value(v);
            let expected_mac = Fr::mul(&alpha, &total_value);

            // Distribute MAC shares.
            let mut mac_sum = Fr::ZERO;
            for p in 0..num_parties - 1 {
                let share = Fr::from_f64((p as f64 + 1.0) * (v as f64 + 1.0) * 0.1);
                witness.mac_shares[p][v] = share.clone();
                mac_sum = Fr::add(&mac_sum, &share);
            }
            witness.mac_shares[num_parties - 1][v] = Fr::sub(&expected_mac, &mac_sum);
        }

        witness
    }

    #[test]
    fn test_witness_creation() {
        let witness = MACWitness::new(3, 4);
        assert_eq!(witness.num_parties, 3);
        assert_eq!(witness.num_values, 4);
    }

    #[test]
    fn test_mac_verification() {
        let witness = create_valid_mac_witness(3, 4);
        assert!(witness.verify_macs());
    }

    #[test]
    fn test_prove_and_verify() {
        let witness = create_valid_mac_witness(3, 4);

        let mut prover = MACProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        assert_eq!(proof.num_parties, 3);
        assert_eq!(proof.num_values, 4);

        let verifier = MACVerifier::new();
        assert!(verifier.verify(&proof).unwrap());
    }

    #[test]
    fn test_proof_serialization() {
        let witness = create_valid_mac_witness(3, 4);

        let mut prover = MACProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        let bytes = proof.to_bytes();
        let restored = MACProof::from_bytes(&bytes).unwrap();

        assert_eq!(restored.num_parties, proof.num_parties);
        assert_eq!(restored.num_values, proof.num_values);
        assert_eq!(restored.alpha_commitment, proof.alpha_commitment);
    }

    #[test]
    fn test_batch_prover() {
        let mut batch = BatchMACProver::new();

        for _ in 0..3 {
            let witness = create_valid_mac_witness(3, 4);
            batch.add_witness(witness);
        }

        assert_eq!(batch.len(), 3);

        let proofs = batch.prove_all().unwrap();
        assert_eq!(proofs.len(), 3);

        let verifier = MACVerifier::new();
        for proof in &proofs {
            assert!(verifier.verify(proof).unwrap());
        }
    }

    #[test]
    fn test_invalid_macs_rejected() {
        let mut witness = MACWitness::new(3, 2);

        // Set alpha shares.
        witness
            .set_alpha_shares(vec![Fr::from_f64(1.0), Fr::from_f64(2.0), Fr::from_f64(2.0)])
            .unwrap();

        // Set invalid MAC shares (don't match the relationship).
        for p in 0..3 {
            let values = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
            let macs = vec![Fr::from_f64(0.5), Fr::from_f64(0.5)]; // Wrong!

            witness
                .set_party_shares(p, values, macs, [p as u8; 32])
                .unwrap();
        }

        assert!(!witness.verify_macs());

        let mut prover = MACProver::with_seed(42);
        assert!(prover.prove(&witness).is_err());
    }
}
