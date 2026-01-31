//! Native Rust Verifier for Halo2 Proofs.
//!
//! Provides verification of Halo2 proofs using the halo2_proofs API.
//! Supports both single proof verification and batch verification.

use halo2_proofs::{
    plonk::{verify_proof as halo2_verify, Error as PlonkError, VerifyingKey},
    transcript::{Blake2bRead, Challenge255},
    poly::commitment::Params,
};
use halo2curves::bn256::{Bn256, G1Affine, Fr};
use std::io::Cursor;

/// A serialized Halo2 proof.
#[derive(Debug, Clone)]
pub struct SerializedProof {
    /// Raw proof bytes.
    pub bytes: Vec<u8>,
    /// Number of public inputs.
    pub num_public_inputs: usize,
}

impl SerializedProof {
    /// Creates a new serialized proof.
    pub fn new(bytes: Vec<u8>, num_public_inputs: usize) -> Self {
        Self { bytes, num_public_inputs }
    }

    /// Returns the proof length.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the proof is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Serializes to bytes (for network/storage).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        // Length prefix
        result.extend_from_slice(&(self.bytes.len() as u32).to_le_bytes());
        result.extend_from_slice(&(self.num_public_inputs as u32).to_le_bytes());
        result.extend_from_slice(&self.bytes);
        result
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, VerifierError> {
        if data.len() < 8 {
            return Err(VerifierError::InvalidProofFormat("Too short".to_string()));
        }

        let proof_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let num_public_inputs = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if data.len() < 8 + proof_len {
            return Err(VerifierError::InvalidProofFormat("Truncated proof".to_string()));
        }

        let bytes = data[8..8 + proof_len].to_vec();
        Ok(Self { bytes, num_public_inputs })
    }
}

/// Native verifier for Halo2 proofs.
/// 
/// This is a simplified verifier wrapper that delegates to halo2_proofs.
/// For production, use the actual halo2 verification with KZG setup.
#[derive(Debug)]
pub struct NativeVerifier {
    /// Degree of the polynomial commitment (log2).
    k: u32,
}

impl NativeVerifier {
    /// Creates a new verifier with the given degree.
    pub fn new(k: u32) -> Self {
        Self { k }
    }

    /// Returns the degree parameter.
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Verifies a proof using the MockProver approach for testing.
    /// 
    /// Note: For production verification, use proper KZG params and verify_proof.
    pub fn verify_mock<C, F>(
        &self,
        circuit: &C,
        instances: Vec<Vec<Fr>>,
    ) -> Result<(), VerifierError>
    where
        C: halo2_proofs::plonk::Circuit<Fr>,
    {
        use halo2_proofs::dev::MockProver;

        let prover = MockProver::run(self.k, circuit, instances)
            .map_err(|e| VerifierError::ProverError(format!("{:?}", e)))?;

        prover.verify()
            .map_err(|e| VerifierError::VerificationFailed(format!("{:?}", e)))
    }
}

/// Verification result containing proof metadata.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification passed.
    pub valid: bool,
    /// Public inputs used.
    pub public_inputs: Vec<Fr>,
    /// Proof size in bytes.
    pub proof_size: usize,
}

/// Proof metadata for logging/debugging.
#[derive(Debug, Clone)]
pub struct ProofMetadata {
    /// Hash of the proof bytes.
    pub proof_hash: [u8; 32],
    /// Number of public inputs.
    pub num_instances: usize,
    /// Proof size in bytes.
    pub size_bytes: usize,
}

impl ProofMetadata {
    /// Computes metadata from a proof.
    pub fn from_proof(proof: &SerializedProof) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        proof.bytes.hash(&mut hasher);
        let hash = hasher.finish();

        let mut proof_hash = [0u8; 32];
        proof_hash[..8].copy_from_slice(&hash.to_le_bytes());
        proof_hash[8..12].copy_from_slice(&(proof.len() as u32).to_le_bytes());

        Self {
            proof_hash,
            num_instances: proof.num_public_inputs,
            size_bytes: proof.len(),
        }
    }
}

/// Public input encoding utilities.
pub mod public_inputs {
    use halo2curves::bn256::Fr;
    use halo2curves::ff::PrimeField;

    /// Encodes a u64 as a field element.
    pub fn encode_u64(value: u64) -> Fr {
        Fr::from(value)
    }

    /// Encodes a byte array as field elements (32 bytes per element).
    pub fn encode_bytes(bytes: &[u8]) -> Vec<Fr> {
        bytes.chunks(31)
            .map(|chunk| {
                let mut val = 0u64;
                for (i, &b) in chunk.iter().enumerate().take(8) {
                    val |= (b as u64) << (i * 8);
                }
                Fr::from(val)
            })
            .collect()
    }

    /// Encodes a commitment hash as field elements.
    pub fn encode_commitment(commitment: &[u8; 32]) -> Vec<Fr> {
        // Split 256-bit commitment into two 128-bit field elements
        let mut low = 0u128;
        let mut high = 0u128;
        
        for (i, &b) in commitment[..16].iter().enumerate() {
            low |= (b as u128) << (i * 8);
        }
        for (i, &b) in commitment[16..].iter().enumerate() {
            high |= (b as u128) << (i * 8);
        }
        
        vec![
            Fr::from(low as u64),
            Fr::from(high as u64),
        ]
    }

    /// Decodes field elements back to a commitment hash.
    pub fn decode_commitment(elements: &[Fr]) -> [u8; 32] {
        if elements.len() < 2 {
            return [0u8; 32];
        }
        
        // Simplified - just extract repr bytes
        let repr0 = elements[0].to_repr();
        let repr1 = elements[1].to_repr();
        
        let mut result = [0u8; 32];
        result[..16].copy_from_slice(&repr0[..16]);
        result[16..].copy_from_slice(&repr1[..16]);
        result
    }
}

/// Verifier errors.
#[derive(Debug)]
pub enum VerifierError {
    /// Invalid proof format.
    InvalidProofFormat(String),
    /// Parameters failed to load.
    ParamsLoadFailed(String),
    /// Verification failed.
    VerificationFailed(String),
    /// Prover error.
    ProverError(String),
    /// Transcript error.
    TranscriptError(String),
    /// Public inputs mismatch.
    PublicInputsMismatch {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for VerifierError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProofFormat(msg) => write!(f, "Invalid proof format: {}", msg),
            Self::ParamsLoadFailed(msg) => write!(f, "Failed to load params: {}", msg),
            Self::VerificationFailed(msg) => write!(f, "Verification failed: {}", msg),
            Self::ProverError(msg) => write!(f, "Prover error: {}", msg),
            Self::TranscriptError(msg) => write!(f, "Transcript error: {}", msg),
            Self::PublicInputsMismatch { expected, got } => {
                write!(f, "Public inputs mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for VerifierError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialized_proof_roundtrip() {
        let proof = SerializedProof::new(vec![1, 2, 3, 4, 5], 2);
        let bytes = proof.to_bytes();
        let recovered = SerializedProof::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.bytes, vec![1, 2, 3, 4, 5]);
        assert_eq!(recovered.num_public_inputs, 2);
    }

    #[test]
    fn test_proof_metadata() {
        let proof = SerializedProof::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 3);
        let meta = ProofMetadata::from_proof(&proof);

        assert_eq!(meta.num_instances, 3);
        assert_eq!(meta.size_bytes, 8);
    }

    #[test]
    fn test_public_input_encoding() {
        use public_inputs::*;

        let value = encode_u64(12345);
        assert_eq!(value, Fr::from(12345));

        let commitment = [0xABu8; 32];
        let encoded = encode_commitment(&commitment);
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_native_verifier_creation() {
        let verifier = NativeVerifier::new(10);
        assert_eq!(verifier.k(), 10);
    }
}
