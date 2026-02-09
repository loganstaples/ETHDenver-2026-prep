//! Keccak256-based Transcript for EVM-compatible Fiat-Shamir.
//!
//! Halo2's default transcript uses Blake2b for challenge derivation, but the
//! EVM only has keccak256 as a native hash. This module provides a Keccak256
//! transcript that produces identical challenges on both the Rust prover
//! and the Solidity verifier, enabling on-chain proof verification.
//!
//! # Challenge Derivation
//!
//! The Solidity verifier computes challenges as:
//! ```solidity
//! bytes32 seed = keccak256(abi.encodePacked(proof, instances));
//! alpha = uint256(seed) % R;
//! beta  = uint256(keccak256(abi.encodePacked(seed, uint256(1)))) % R;
//! gamma = uint256(keccak256(abi.encodePacked(seed, uint256(2)))) % R;
//! ```
//!
//! This transcript mirrors that logic:
//! 1. Points and scalars are absorbed in big-endian (EVM) encoding
//! 2. Challenges are squeezed via keccak256 of the running state
//! 3. Sequential challenges use domain separation (counter)

use halo2curves::bn256::{Fr, G1Affine, Fq};
use halo2curves::ff::PrimeField;
use halo2curves::CurveAffine;
use sha3::{Digest, Keccak256};
use std::io::{self, Write};

use super::format_spec::{fr_to_evm_bytes, fq_to_evm_bytes};

/// Keccak256-based transcript writer for EVM-compatible Fiat-Shamir.
///
/// Mirrors the Halo2 `TranscriptWrite` interface but uses keccak256
/// instead of Blake2b, enabling challenge alignment with Solidity.
#[derive(Debug, Clone)]
pub struct Keccak256Write<W: Write> {
    /// The output stream for proof bytes.
    writer: W,
    /// Running hash state: accumulated absorbed data.
    state: Vec<u8>,
    /// Counter for domain-separated sequential challenges.
    squeeze_count: u64,
}

impl<W: Write> Keccak256Write<W> {
    /// Initializes a new Keccak256 transcript writer.
    pub fn init(writer: W) -> Self {
        Self {
            writer,
            state: Vec::new(),
            squeeze_count: 0,
        }
    }

    /// Consumes the transcript and returns the inner writer.
    pub fn finalize(self) -> W {
        self.writer
    }

    /// Returns a reference to the inner writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Absorbs a G1 point into the transcript state (EVM big-endian encoding).
    ///
    /// This is the equivalent of Halo2's `common_point`.
    pub fn common_point(&mut self, point: &G1Affine) -> io::Result<()> {
        let x_bytes = fq_to_evm_bytes(&point.x);
        let y_bytes = fq_to_evm_bytes(&point.y);
        self.state.extend_from_slice(&x_bytes);
        self.state.extend_from_slice(&y_bytes);
        Ok(())
    }

    /// Absorbs a scalar into the transcript state (EVM big-endian encoding).
    ///
    /// This is the equivalent of Halo2's `common_scalar`.
    pub fn common_scalar(&mut self, scalar: &Fr) -> io::Result<()> {
        let bytes = fr_to_evm_bytes(scalar);
        self.state.extend_from_slice(&bytes);
        Ok(())
    }

    /// Writes a G1 point to both the transcript state and the output stream.
    ///
    /// This is the equivalent of Halo2's `TranscriptWrite::write_point`.
    /// The point is written to the output in Halo2's native little-endian
    /// format (for transcript compatibility), but absorbed in big-endian
    /// (for Solidity challenge alignment).
    pub fn write_point(&mut self, point: &G1Affine) -> io::Result<()> {
        self.common_point(point)?;
        // Write to output in Fq repr (little-endian, native halo2 format)
        let x_repr = point.x.to_repr();
        let y_repr = point.y.to_repr();
        self.writer.write_all(x_repr.as_ref())?;
        self.writer.write_all(y_repr.as_ref())?;
        Ok(())
    }

    /// Writes a scalar to both the transcript state and the output stream.
    ///
    /// This is the equivalent of Halo2's `TranscriptWrite::write_scalar`.
    pub fn write_scalar(&mut self, scalar: &Fr) -> io::Result<()> {
        self.common_scalar(scalar)?;
        // Write to output in Fr repr (little-endian, native halo2 format)
        let repr = scalar.to_repr();
        self.writer.write_all(repr.as_ref())?;
        Ok(())
    }

    /// Squeezes a challenge scalar from the transcript.
    ///
    /// Matches the Solidity pattern:
    /// ```solidity
    /// seed = keccak256(state);              // first challenge
    /// keccak256(abi.encodePacked(seed, n))  // subsequent challenges
    /// ```
    ///
    /// The result is reduced modulo the BN254 scalar field order R.
    pub fn squeeze_challenge(&mut self) -> Fr {
        let challenge_bytes = if self.squeeze_count == 0 {
            // First challenge: hash the entire accumulated state
            let hash = Keccak256::digest(&self.state);
            hash.to_vec()
        } else {
            // Subsequent challenges: domain-separated from the seed
            let seed = Keccak256::digest(&self.state);
            let mut input = Vec::with_capacity(64);
            input.extend_from_slice(&seed);
            // Encode counter as big-endian uint256 (matching Solidity abi.encodePacked)
            let mut counter = [0u8; 32];
            counter[24..].copy_from_slice(&self.squeeze_count.to_be_bytes());
            input.extend_from_slice(&counter);
            let hash = Keccak256::digest(&input);
            hash.to_vec()
        };

        self.squeeze_count += 1;

        // Convert big-endian hash to Fr (mod R)
        // This matches Solidity: uint256(hash) % R
        hash_to_fr(&challenge_bytes)
    }

    /// Resets the squeeze counter (call between protocol phases).
    pub fn reset_challenge_counter(&mut self) {
        self.squeeze_count = 0;
    }
}

/// Keccak256-based transcript reader for EVM-compatible Fiat-Shamir verification.
///
/// This is the verifier-side counterpart to `Keccak256Write`.
#[derive(Debug, Clone)]
pub struct Keccak256Read<R: io::Read> {
    /// The input stream for proof bytes.
    reader: R,
    /// Running hash state.
    state: Vec<u8>,
    /// Counter for domain-separated challenges.
    squeeze_count: u64,
}

impl<R: io::Read> Keccak256Read<R> {
    /// Initializes a new Keccak256 transcript reader.
    pub fn init(reader: R) -> Self {
        Self {
            reader,
            state: Vec::new(),
            squeeze_count: 0,
        }
    }

    /// Absorbs a G1 point into the transcript state.
    pub fn common_point(&mut self, point: &G1Affine) -> io::Result<()> {
        let x_bytes = fq_to_evm_bytes(&point.x);
        let y_bytes = fq_to_evm_bytes(&point.y);
        self.state.extend_from_slice(&x_bytes);
        self.state.extend_from_slice(&y_bytes);
        Ok(())
    }

    /// Absorbs a scalar into the transcript state.
    pub fn common_scalar(&mut self, scalar: &Fr) -> io::Result<()> {
        let bytes = fr_to_evm_bytes(scalar);
        self.state.extend_from_slice(&bytes);
        Ok(())
    }

    /// Reads a G1 point from the proof stream and absorbs it.
    pub fn read_point(&mut self) -> io::Result<G1Affine> {
        let mut x_bytes = [0u8; 32];
        let mut y_bytes = [0u8; 32];
        self.reader.read_exact(&mut x_bytes)?;
        self.reader.read_exact(&mut y_bytes)?;

        let x = Fq::from_repr(x_bytes.into());
        let y = Fq::from_repr(y_bytes.into());

        if bool::from(x.is_none()) || bool::from(y.is_none()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid field element in proof",
            ));
        }

        // Safety: x and y are guaranteed Some by the is_none() checks above
        let (x_val, y_val) = (x.unwrap(), y.unwrap());
        let point = G1Affine::from_xy(x_val, y_val);
        let point = Option::from(point).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Point not on curve")
        })?;
        self.common_point(&point)?;
        Ok(point)
    }

    /// Reads a scalar from the proof stream and absorbs it.
    pub fn read_scalar(&mut self) -> io::Result<Fr> {
        let mut bytes = [0u8; 32];
        self.reader.read_exact(&mut bytes)?;

        let scalar = Option::from(Fr::from_repr(bytes.into())).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Invalid scalar in proof")
        })?;
        self.common_scalar(&scalar)?;
        Ok(scalar)
    }

    /// Squeezes a challenge scalar (identical logic to Keccak256Write).
    pub fn squeeze_challenge(&mut self) -> Fr {
        let challenge_bytes = if self.squeeze_count == 0 {
            let hash = Keccak256::digest(&self.state);
            hash.to_vec()
        } else {
            let seed = Keccak256::digest(&self.state);
            let mut input = Vec::with_capacity(64);
            input.extend_from_slice(&seed);
            let mut counter = [0u8; 32];
            counter[24..].copy_from_slice(&self.squeeze_count.to_be_bytes());
            input.extend_from_slice(&counter);
            let hash = Keccak256::digest(&input);
            hash.to_vec()
        };

        self.squeeze_count += 1;
        hash_to_fr(&challenge_bytes)
    }
}

/// Converts a 32-byte big-endian hash to Fr by reduction modulo R.
///
/// This matches the Solidity pattern: `uint256(hash) % R`
///
/// Uses Fr field arithmetic (Horner's method over u64 limbs) to handle
/// values that may exceed the field modulus R.
pub(crate) fn hash_to_fr(hash_bytes: &[u8]) -> Fr {
    // Interpret the 32-byte big-endian hash as 4 big-endian u64 limbs
    // hash_bytes[0..8] is the most significant limb
    let mut padded = [0u8; 32];
    let start = 32 - hash_bytes.len().min(32);
    padded[start..].copy_from_slice(&hash_bytes[..hash_bytes.len().min(32)]);

    let limb3 = u64::from_be_bytes(padded[0..8].try_into().unwrap());   // most significant
    let limb2 = u64::from_be_bytes(padded[8..16].try_into().unwrap());
    let limb1 = u64::from_be_bytes(padded[16..24].try_into().unwrap());
    let limb0 = u64::from_be_bytes(padded[24..32].try_into().unwrap()); // least significant

    // Compute: (((limb3 * 2^64 + limb2) * 2^64 + limb1) * 2^64 + limb0) mod R
    // Using Fr arithmetic which handles modular reduction automatically.
    let shift = Fr::from(1u64 << 32) * Fr::from(1u64 << 32); // 2^64 as Fr
    let mut result = Fr::from(limb3);
    result = result * shift + Fr::from(limb2);
    result = result * shift + Fr::from(limb1);
    result = result * shift + Fr::from(limb0);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::group::Curve;

    #[test]
    fn test_keccak_transcript_challenge_determinism() {
        let mut t1 = Keccak256Write::init(Vec::new());
        let mut t2 = Keccak256Write::init(Vec::new());

        let g = G1Affine::generator();
        t1.write_point(&g).unwrap();
        t2.write_point(&g).unwrap();

        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();
        assert_eq!(c1, c2, "Same inputs must produce same challenge");
    }

    #[test]
    fn test_keccak_transcript_different_inputs_different_challenges() {
        let g = G1Affine::generator();
        let p = (g * Fr::from(42u64)).to_affine();

        let mut t1 = Keccak256Write::init(Vec::new());
        t1.write_point(&g).unwrap();
        let c1 = t1.squeeze_challenge();

        let mut t2 = Keccak256Write::init(Vec::new());
        t2.write_point(&p).unwrap();
        let c2 = t2.squeeze_challenge();

        assert_ne!(c1, c2, "Different inputs must produce different challenges");
    }

    #[test]
    fn test_keccak_transcript_sequential_challenges() {
        let mut t = Keccak256Write::init(Vec::new());
        let g = G1Affine::generator();
        t.write_point(&g).unwrap();

        let alpha = t.squeeze_challenge();
        let beta = t.squeeze_challenge();
        let gamma = t.squeeze_challenge();

        // All three should be different
        assert_ne!(alpha, beta);
        assert_ne!(beta, gamma);
        assert_ne!(alpha, gamma);

        // All should be non-zero (astronomically unlikely to be zero)
        assert_ne!(alpha, Fr::zero());
        assert_ne!(beta, Fr::zero());
        assert_ne!(gamma, Fr::zero());
    }

    #[test]
    fn test_keccak_transcript_prover_verifier_agreement() {
        let g = G1Affine::generator();
        let p = (g * Fr::from(123u64)).to_affine();
        let s = Fr::from(456u64);

        // Prover side
        let mut prover_transcript = Keccak256Write::init(Vec::new());
        prover_transcript.write_point(&g).unwrap();
        prover_transcript.write_point(&p).unwrap();
        prover_transcript.write_scalar(&s).unwrap();
        let prover_challenge = prover_transcript.squeeze_challenge();
        let proof_bytes = prover_transcript.finalize();

        // Verifier side
        let mut verifier_transcript = Keccak256Read::init(proof_bytes.as_slice());
        let read_g = verifier_transcript.read_point().unwrap();
        let read_p = verifier_transcript.read_point().unwrap();
        let read_s = verifier_transcript.read_scalar().unwrap();

        assert_eq!(read_g, g);
        assert_eq!(read_p, p);
        assert_eq!(read_s, s);

        let verifier_challenge = verifier_transcript.squeeze_challenge();
        assert_eq!(
            prover_challenge, verifier_challenge,
            "Prover and verifier must derive identical challenges"
        );
    }

    #[test]
    fn test_keccak_transcript_matches_solidity_pattern() {
        // Verify the challenge derivation matches the Solidity pattern:
        //   seed = keccak256(data)
        //   alpha = uint256(seed) % R
        //   beta = uint256(keccak256(seed || uint256(1))) % R

        let g = G1Affine::generator();
        let mut t = Keccak256Write::init(Vec::new());
        t.common_point(&g).unwrap();

        // Manually compute what Solidity would do
        let x_bytes = fq_to_evm_bytes(&g.x);
        let y_bytes = fq_to_evm_bytes(&g.y);
        let mut data = Vec::new();
        data.extend_from_slice(&x_bytes);
        data.extend_from_slice(&y_bytes);

        let seed = Keccak256::digest(&data);
        let alpha_manual = hash_to_fr(&seed);

        let alpha = t.squeeze_challenge();
        assert_eq!(alpha, alpha_manual, "First challenge must match keccak256(data) % R");

        // Second challenge
        let mut input2 = Vec::with_capacity(64);
        input2.extend_from_slice(&seed);
        let mut counter = [0u8; 32];
        counter[24..].copy_from_slice(&1u64.to_be_bytes());
        input2.extend_from_slice(&counter);
        let hash2 = Keccak256::digest(&input2);
        let beta_manual = hash_to_fr(&hash2);

        let beta = t.squeeze_challenge();
        assert_eq!(beta, beta_manual, "Second challenge must match domain-separated keccak");
    }

    #[test]
    fn test_keccak_transcript_output_bytes() {
        let mut t = Keccak256Write::init(Vec::new());
        let g = G1Affine::generator();
        t.write_point(&g).unwrap();
        t.write_scalar(&Fr::from(42u64)).unwrap();

        let output = t.finalize();
        // Point = 64 bytes (LE Fq x + y), Scalar = 32 bytes (LE Fr)
        assert_eq!(output.len(), 64 + 32);
    }
}
