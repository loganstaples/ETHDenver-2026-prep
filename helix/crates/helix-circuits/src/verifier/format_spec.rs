//! EVM Proof Format Specification
//!
//! This module documents the exact byte format expected by the Halo2Verifier.sol
//! contract for on-chain proof verification. This is critical infrastructure for
//! ensuring Rust-generated proofs are compatible with the Solidity verifier.
//!
//! # Proof Byte Layout
//!
//! The Halo2Verifier contract expects proof bytes structured as follows:
//!
//! ```text
//! Offset  Size    Content
//! ------  ----    -------
//! 0       64      Advice commitment point 0 (C0): (x: u256, y: u256)
//! 64      64      Advice commitment point 1 (C1): (x: u256, y: u256)
//! 128     64      Advice commitment point 2 (C2): (x: u256, y: u256)
//! 192     64      Opening proof point W: (x: u256, y: u256)
//! 256     64      Opening proof point W': (x: u256, y: u256)
//! ------  ----    -------
//! Total   320     Minimum proof size
//! ```
//!
//! Each coordinate is a 32-byte big-endian representation of a BN254 scalar field element.
//! All points must be valid points on the BN254 G1 curve (y² = x³ + 3).
//!
//! # Public Inputs Layout
//!
//! The MLTrainingStepV2 circuit produces 7 public inputs:
//!
//! ```text
//! Index   Content                 Description
//! -----   -------                 -----------
//! 0       old_state_hash_lo       Lower 128 bits of SHA256(old_weights)
//! 1       old_state_hash_hi       Upper 128 bits of SHA256(old_weights)
//! 2       new_state_hash_lo       Lower 128 bits of SHA256(new_weights)
//! 3       new_state_hash_hi       Upper 128 bits of SHA256(new_weights)
//! 4       loss                    Quantized training loss value
//! 5       error_bound             Accumulated error bound for this step
//! 6       step_number             Training step counter (0-indexed)
//! ```
//!
//! Each public input is a uint256 value that must be less than the scalar field order R.
//!
//! # Commitment Reconstruction
//!
//! The contract reconstructs model commitments using:
//!
//! ```solidity
//! function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
//!     return uint256(keccak256(abi.encodePacked(lo, hi)));
//! }
//! ```
//!
//! This means:
//! - `lo` and `hi` are each encoded as 32-byte big-endian values
//! - The hash is computed over exactly 64 bytes (lo || hi)
//! - The result is the full uint256 of the keccak256 hash
//!
//! # Field Element Encoding
//!
//! BN254 scalar field order:
//! R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//!
//! BN254 base field prime:
//! P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//!
//! All field elements must be:
//! 1. Less than R (for scalars/public inputs)
//! 2. Less than P (for curve point coordinates)
//! 3. Encoded as 32-byte big-endian values
//!
//! # Verification Flow
//!
//! The contract verifies proofs using the following steps:
//!
//! 1. **Parse proof bytes**: Extract 3 advice commits + 2 opening proofs
//! 2. **Validate curve membership**: All 5 points must satisfy y² = x³ + 3 (mod P)
//! 3. **Compute Fiat-Shamir challenges**: α, β, γ = H(proof || publicInputs)
//! 4. **Combine commitments**: P = Σ(αⁱ · Cᵢ)
//! 5. **Compute evaluation point**: v = [β]·G₁
//! 6. **Form pairing points**:
//!    - A = P - v + γ·W'
//!    - B = W + γ·W'
//! 7. **Pairing check**: e(A, -[1]₂) · e(B, [s]₂) = 1

use halo2curves::bn256::{Fr, G1Affine, Fq};
use halo2curves::ff::{PrimeField, Field};
use halo2curves::group::prime::PrimeCurveAffine;
use halo2curves::CurveAffine;

/// Minimum proof size required by Halo2Verifier.sol
pub const MIN_PROOF_SIZE: usize = 320;

/// Number of advice commitment points in the proof
pub const NUM_ADVICE_COMMITS: usize = 3;

/// Size of a G1 point in bytes (2 × 32-byte coordinates)
pub const G1_POINT_SIZE: usize = 64;

/// Size of a G2 point in bytes (4 × 32-byte coordinates for Fp2)
pub const G2_POINT_SIZE: usize = 128;

/// Size of a scalar field element in bytes
pub const SCALAR_SIZE: usize = 32;

/// Number of public inputs for MLTrainingStepV2
pub const NUM_PUBLIC_INPUTS: usize = 7;

/// BN254 scalar field order as bytes (big-endian)
pub const SCALAR_FIELD_ORDER: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29,
    0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x97, 0x81, 0x6a, 0x91, 0x68, 0x71, 0xca, 0x8d,
    0x3c, 0x20, 0x8c, 0x16, 0xd8, 0x7c, 0xfd, 0x47,
];

/// BN254 base field prime as bytes (big-endian)
pub const BASE_FIELD_PRIME: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29,
    0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d,
    0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91,
    0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// Proof format specification for EVM verification
#[derive(Debug, Clone)]
pub struct EvmProofFormat {
    /// Three advice commitment points (C0, C1, C2)
    pub advice_commits: [(G1Affine, G1Affine, G1Affine); 1],
    /// Opening proof W point
    pub w: G1Affine,
    /// Opening proof W' point
    pub w_prime: G1Affine,
}

/// Public input specification for EVM verification
#[derive(Debug, Clone)]
pub struct EvmPublicInputs {
    /// Old state hash split into (lo, hi) 128-bit halves
    pub old_state_hash: (Fr, Fr),
    /// New state hash split into (lo, hi) 128-bit halves
    pub new_state_hash: (Fr, Fr),
    /// Computed loss value
    pub loss: Fr,
    /// Accumulated error bound
    pub error_bound: Fr,
    /// Training step number
    pub step_number: Fr,
}

impl EvmPublicInputs {
    /// Creates public inputs from the 7-element array
    pub fn from_array(inputs: &[Fr; NUM_PUBLIC_INPUTS]) -> Self {
        Self {
            old_state_hash: (inputs[0], inputs[1]),
            new_state_hash: (inputs[2], inputs[3]),
            loss: inputs[4],
            error_bound: inputs[5],
            step_number: inputs[6],
        }
    }

    /// Converts to the 7-element array format
    pub fn to_array(&self) -> [Fr; NUM_PUBLIC_INPUTS] {
        [
            self.old_state_hash.0,
            self.old_state_hash.1,
            self.new_state_hash.0,
            self.new_state_hash.1,
            self.loss,
            self.error_bound,
            self.step_number,
        ]
    }

    /// Encodes to EVM-compatible uint256 array (big-endian bytes)
    pub fn to_evm_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(NUM_PUBLIC_INPUTS * SCALAR_SIZE);
        for fr in self.to_array() {
            result.extend(fr_to_evm_bytes(&fr));
        }
        result
    }
}

/// Proof structure breakdown for debugging and validation
#[derive(Debug, Clone)]
pub struct ProofStructure {
    /// Total proof size in bytes
    pub total_size: usize,
    /// Advice commitment section: bytes 0-191
    pub advice_section: ProofSection,
    /// Opening proof section: bytes 192-319
    pub opening_section: ProofSection,
    /// Optional additional data after byte 320
    pub extra_data: Option<Vec<u8>>,
}

/// A section of the proof
#[derive(Debug, Clone)]
pub struct ProofSection {
    /// Start offset in bytes
    pub offset: usize,
    /// Length in bytes
    pub length: usize,
    /// Description
    pub description: String,
}

impl ProofStructure {
    /// Parses a proof into its structural components
    pub fn parse(proof_bytes: &[u8]) -> Result<Self, ProofFormatError> {
        if proof_bytes.len() < MIN_PROOF_SIZE {
            return Err(ProofFormatError::TooShort {
                got: proof_bytes.len(),
                min: MIN_PROOF_SIZE,
            });
        }

        let advice_section = ProofSection {
            offset: 0,
            length: NUM_ADVICE_COMMITS * G1_POINT_SIZE,
            description: format!("{} advice commitment points", NUM_ADVICE_COMMITS),
        };

        let opening_section = ProofSection {
            offset: NUM_ADVICE_COMMITS * G1_POINT_SIZE,
            length: 2 * G1_POINT_SIZE,
            description: "Opening proof points (W, W')".to_string(),
        };

        let extra_data = if proof_bytes.len() > MIN_PROOF_SIZE {
            Some(proof_bytes[MIN_PROOF_SIZE..].to_vec())
        } else {
            None
        };

        Ok(Self {
            total_size: proof_bytes.len(),
            advice_section,
            opening_section,
            extra_data,
        })
    }

    /// Returns a formatted breakdown of the proof structure
    pub fn dump(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Proof Structure Breakdown ===\n");
        output.push_str(&format!("Total size: {} bytes\n", self.total_size));
        output.push_str(&format!(
            "\nAdvice Commitments: bytes {}-{} ({} bytes)\n",
            self.advice_section.offset,
            self.advice_section.offset + self.advice_section.length - 1,
            self.advice_section.length
        ));
        output.push_str(&format!("  {}\n", self.advice_section.description));

        for i in 0..NUM_ADVICE_COMMITS {
            let start = i * G1_POINT_SIZE;
            output.push_str(&format!(
                "  Point C{}: bytes {}-{}\n",
                i,
                start,
                start + G1_POINT_SIZE - 1
            ));
            output.push_str(&format!("    x: bytes {}-{}\n", start, start + 31));
            output.push_str(&format!("    y: bytes {}-{}\n", start + 32, start + 63));
        }

        output.push_str(&format!(
            "\nOpening Proofs: bytes {}-{} ({} bytes)\n",
            self.opening_section.offset,
            self.opening_section.offset + self.opening_section.length - 1,
            self.opening_section.length
        ));
        output.push_str(&format!("  {}\n", self.opening_section.description));
        output.push_str(&format!(
            "  Point W:  bytes {}-{}\n",
            self.opening_section.offset,
            self.opening_section.offset + G1_POINT_SIZE - 1
        ));
        output.push_str(&format!(
            "  Point W': bytes {}-{}\n",
            self.opening_section.offset + G1_POINT_SIZE,
            self.opening_section.offset + 2 * G1_POINT_SIZE - 1
        ));

        if let Some(ref extra) = self.extra_data {
            output.push_str(&format!(
                "\nExtra data: {} bytes (starting at byte {})\n",
                extra.len(),
                MIN_PROOF_SIZE
            ));
        }

        output
    }
}

/// Errors that can occur during proof format validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofFormatError {
    /// Proof bytes are too short
    TooShort { got: usize, min: usize },
    /// Point is not on the BN254 curve
    PointNotOnCurve { point_index: usize, x: [u8; 32], y: [u8; 32] },
    /// Field element exceeds the field order
    FieldOverflow { index: usize, value: [u8; 32] },
    /// Invalid public input count
    InvalidPublicInputCount { got: usize, expected: usize },
    /// Commitment mismatch with expected value
    CommitmentMismatch { expected: [u8; 32], got: [u8; 32] },
}

impl std::fmt::Display for ProofFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { got, min } => {
                write!(f, "Proof too short: got {} bytes, minimum is {}", got, min)
            }
            Self::PointNotOnCurve { point_index, x, y } => {
                write!(
                    f,
                    "Point {} not on curve: x={}, y={}",
                    point_index,
                    hex::encode(x),
                    hex::encode(y)
                )
            }
            Self::FieldOverflow { index, value } => {
                write!(
                    f,
                    "Field element at index {} exceeds field order: {}",
                    index,
                    hex::encode(value)
                )
            }
            Self::InvalidPublicInputCount { got, expected } => {
                write!(
                    f,
                    "Invalid public input count: got {}, expected {}",
                    got, expected
                )
            }
            Self::CommitmentMismatch { expected, got } => {
                write!(
                    f,
                    "Commitment mismatch: expected {}, got {}",
                    hex::encode(expected),
                    hex::encode(got)
                )
            }
        }
    }
}

impl std::error::Error for ProofFormatError {}

/// Converts a BN254 Fr element to 32-byte big-endian EVM format
pub fn fr_to_evm_bytes(fr: &Fr) -> [u8; 32] {
    let repr = fr.to_repr();
    let mut bytes = [0u8; 32];
    // Fr::to_repr() returns little-endian, EVM expects big-endian
    for (i, b) in repr.as_ref().iter().enumerate() {
        bytes[31 - i] = *b;
    }
    bytes
}

/// Converts 32-byte big-endian EVM bytes to BN254 Fr element
pub fn evm_bytes_to_fr(bytes: &[u8; 32]) -> Option<Fr> {
    let mut le_bytes = [0u8; 32];
    for (i, b) in bytes.iter().enumerate() {
        le_bytes[31 - i] = *b;
    }
    Fr::from_repr(le_bytes.into()).into()
}

/// Converts a BN254 Fq element to 32-byte big-endian EVM format
pub fn fq_to_evm_bytes(fq: &Fq) -> [u8; 32] {
    let repr = fq.to_repr();
    let mut bytes = [0u8; 32];
    // Fq::to_repr() returns little-endian, EVM expects big-endian
    for (i, b) in repr.as_ref().iter().enumerate() {
        bytes[31 - i] = *b;
    }
    bytes
}

/// Converts 32-byte big-endian EVM bytes to BN254 Fq element
pub fn evm_bytes_to_fq(bytes: &[u8; 32]) -> Option<Fq> {
    let mut le_bytes = [0u8; 32];
    for (i, b) in bytes.iter().enumerate() {
        le_bytes[31 - i] = *b;
    }
    Fq::from_repr(le_bytes.into()).into()
}

/// Converts a G1Affine point to 64-byte big-endian EVM format
pub fn g1_to_evm_bytes(point: &G1Affine) -> [u8; G1_POINT_SIZE] {
    let mut bytes = [0u8; G1_POINT_SIZE];
    let x_bytes = fq_to_evm_bytes(&point.x);
    let y_bytes = fq_to_evm_bytes(&point.y);
    bytes[..32].copy_from_slice(&x_bytes);
    bytes[32..].copy_from_slice(&y_bytes);
    bytes
}

/// Converts 64-byte big-endian EVM bytes to G1Affine point
pub fn evm_bytes_to_g1(bytes: &[u8; G1_POINT_SIZE]) -> Option<G1Affine> {
    let x_bytes: [u8; 32] = bytes[..32].try_into().ok()?;
    let y_bytes: [u8; 32] = bytes[32..].try_into().ok()?;

    let x = evm_bytes_to_fq(&x_bytes)?;
    let y = evm_bytes_to_fq(&y_bytes)?;

    // Check if point at infinity (0, 0)
    let x_is_zero: bool = bool::from(x.is_zero());
    let y_is_zero: bool = bool::from(y.is_zero());
    if x_is_zero && y_is_zero {
        return Some(G1Affine::identity());
    }

    // Construct the point using CurveAffine trait
    // from_xy returns CtOption<Self>, need to convert to Option
    let ct_point = G1Affine::from_xy(x, y);
    let is_valid: bool = bool::from(ct_point.is_some());
    if is_valid {
        Some(ct_point.unwrap())
    } else {
        None
    }
}

/// Validates that proof bytes conform to the expected EVM format
pub fn validate_proof_format(proof_bytes: &[u8]) -> Result<(), ProofFormatError> {
    // Check minimum length
    if proof_bytes.len() < MIN_PROOF_SIZE {
        return Err(ProofFormatError::TooShort {
            got: proof_bytes.len(),
            min: MIN_PROOF_SIZE,
        });
    }

    // Validate all 5 G1 points (3 advice + 2 opening)
    for i in 0..5 {
        let offset = i * G1_POINT_SIZE;
        let x_bytes: [u8; 32] = proof_bytes[offset..offset + 32]
            .try_into()
            .expect("slice length is correct");
        let y_bytes: [u8; 32] = proof_bytes[offset + 32..offset + 64]
            .try_into()
            .expect("slice length is correct");

        // Check if point is at infinity (0, 0) which is valid
        if x_bytes == [0u8; 32] && y_bytes == [0u8; 32] {
            continue;
        }

        // Validate point is on the curve
        let point_bytes: [u8; G1_POINT_SIZE] = proof_bytes[offset..offset + G1_POINT_SIZE]
            .try_into()
            .expect("slice length is correct");
        if evm_bytes_to_g1(&point_bytes).is_none() {
            return Err(ProofFormatError::PointNotOnCurve {
                point_index: i,
                x: x_bytes,
                y: y_bytes,
            });
        }
    }

    Ok(())
}

/// Validates public inputs conform to the expected format
pub fn validate_public_inputs(inputs: &[Fr]) -> Result<(), ProofFormatError> {
    if inputs.len() != NUM_PUBLIC_INPUTS {
        return Err(ProofFormatError::InvalidPublicInputCount {
            got: inputs.len(),
            expected: NUM_PUBLIC_INPUTS,
        });
    }

    // All Fr elements are automatically within the field, so no overflow check needed
    Ok(())
}

/// Computes the contract's _hashPair commitment from lo/hi halves
///
/// This matches the Solidity:
/// ```solidity
/// function _hashPair(uint256 lo, uint256 hi) internal pure returns (uint256) {
///     return uint256(keccak256(abi.encodePacked(lo, hi)));
/// }
/// ```
pub fn compute_hash_pair(lo: &Fr, hi: &Fr) -> [u8; 32] {
    use sha3::{Digest, Keccak256};

    let lo_bytes = fr_to_evm_bytes(lo);
    let hi_bytes = fr_to_evm_bytes(hi);

    let mut hasher = Keccak256::new();
    hasher.update(&lo_bytes);
    hasher.update(&hi_bytes);

    hasher.finalize().into()
}

/// Verifies that the lo/hi split of a state hash matches the expected commitment
pub fn verify_commitment(
    lo: &Fr,
    hi: &Fr,
    expected_commitment: &[u8; 32],
) -> Result<(), ProofFormatError> {
    let computed = compute_hash_pair(lo, hi);
    if computed != *expected_commitment {
        return Err(ProofFormatError::CommitmentMismatch {
            expected: *expected_commitment,
            got: computed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::ff::Field;

    #[test]
    fn test_fr_roundtrip() {
        let fr = Fr::from(12345u64);
        let bytes = fr_to_evm_bytes(&fr);
        let recovered = evm_bytes_to_fr(&bytes).unwrap();
        assert_eq!(fr, recovered);
    }

    #[test]
    fn test_fr_big_endian() {
        let fr = Fr::from(0x0102030405060708u64);
        let bytes = fr_to_evm_bytes(&fr);
        // Big-endian: most significant bytes first
        assert_eq!(bytes[24..], [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        // Leading bytes should be zero
        assert!(bytes[..24].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_proof_structure_parse() {
        // Create a 320-byte proof
        let proof = vec![0u8; MIN_PROOF_SIZE];
        let structure = ProofStructure::parse(&proof).unwrap();

        assert_eq!(structure.total_size, MIN_PROOF_SIZE);
        assert_eq!(structure.advice_section.offset, 0);
        assert_eq!(structure.advice_section.length, 192);
        assert_eq!(structure.opening_section.offset, 192);
        assert_eq!(structure.opening_section.length, 128);
        assert!(structure.extra_data.is_none());
    }

    #[test]
    fn test_proof_too_short() {
        let proof = vec![0u8; 100];
        let result = validate_proof_format(&proof);
        assert!(matches!(
            result,
            Err(ProofFormatError::TooShort { got: 100, min: 320 })
        ));
    }

    #[test]
    fn test_public_inputs_conversion() {
        let inputs = EvmPublicInputs {
            old_state_hash: (Fr::from(1u64), Fr::from(2u64)),
            new_state_hash: (Fr::from(3u64), Fr::from(4u64)),
            loss: Fr::from(100u64),
            error_bound: Fr::from(10u64),
            step_number: Fr::from(42u64),
        };

        let array = inputs.to_array();
        assert_eq!(array[0], Fr::from(1u64));
        assert_eq!(array[6], Fr::from(42u64));

        let recovered = EvmPublicInputs::from_array(&array);
        assert_eq!(recovered.step_number, inputs.step_number);
    }

    #[test]
    fn test_hash_pair_consistency() {
        // Test with known values
        let lo = Fr::from(1u64);
        let hi = Fr::from(2u64);

        let hash = compute_hash_pair(&lo, &hi);

        // The hash should be 32 bytes
        assert_eq!(hash.len(), 32);

        // Verify it's deterministic
        let hash2 = compute_hash_pair(&lo, &hi);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_validate_public_inputs() {
        // Correct number of inputs
        let inputs: Vec<Fr> = (0..7).map(|i| Fr::from(i as u64)).collect();
        assert!(validate_public_inputs(&inputs).is_ok());

        // Wrong number of inputs
        let short: Vec<Fr> = (0..6).map(|i| Fr::from(i as u64)).collect();
        assert!(matches!(
            validate_public_inputs(&short),
            Err(ProofFormatError::InvalidPublicInputCount { got: 6, expected: 7 })
        ));
    }

    #[test]
    fn test_structure_dump() {
        let proof = vec![0u8; MIN_PROOF_SIZE + 64];
        let structure = ProofStructure::parse(&proof).unwrap();
        let dump = structure.dump();

        assert!(dump.contains("Total size: 384 bytes"));
        assert!(dump.contains("Advice Commitments"));
        assert!(dump.contains("Opening Proofs"));
        assert!(dump.contains("Extra data: 64 bytes"));
    }
}
