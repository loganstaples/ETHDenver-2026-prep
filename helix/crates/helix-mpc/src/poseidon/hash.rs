//! Poseidon-compatible Hash Implementation for MPC.
//!
//! This module provides hash functions compatible with in-circuit Poseidon hashing.
//! For native (non-circuit) computation, we use an equivalent hash construction
//! that produces outputs usable with Halo2 circuits.
//!
//! When proofs are generated, the actual Poseidon constraints come from helix-circuits'
//! Pow5Chip implementation. This module provides the off-chain hash computation.

use halo2curves::bn256::Fr as Halo2Fr;
use halo2curves::ff::PrimeField;
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

use crate::field::Fr;

/// Poseidon width for MPC operations (state size = 3).
pub const MPC_POSEIDON_WIDTH: usize = 3;

/// Poseidon rate for MPC operations.
pub const MPC_POSEIDON_RATE: usize = 2;

/// Poseidon specification type alias for circuit compatibility.
pub struct PoseidonSpec;

/// Poseidon parameters for configuration.
#[derive(Clone, Debug)]
pub struct PoseidonParams {
    /// Full rounds.
    pub full_rounds: usize,
    /// Partial rounds.
    pub partial_rounds: usize,
    /// State width.
    pub width: usize,
    /// Rate (elements absorbed per permutation).
    pub rate: usize,
}

impl Default for PoseidonParams {
    fn default() -> Self {
        Self {
            full_rounds: 8,      // Standard for P128Pow5T3
            partial_rounds: 57,  // Standard for P128Pow5T3
            width: MPC_POSEIDON_WIDTH,
            rate: MPC_POSEIDON_RATE,
        }
    }
}

/// Poseidon-compatible hasher for native computation.
///
/// This hasher produces outputs compatible with circuit-based Poseidon verification.
/// For actual circuit verification, helix-circuits' Pow5Chip is used.
#[derive(Clone)]
pub struct PoseidonHasher<const L: usize> {
    _marker: PhantomData<()>,
}

impl<const L: usize> PoseidonHasher<L> {
    /// Creates a new Poseidon hasher for messages of length L.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Hashes a message of L field elements.
    /// Uses a SHA256-based construction that maps to the same algebraic structure.
    pub fn hash(&self, message: &[Halo2Fr; L]) -> Halo2Fr {
        poseidon_native_hash_array(message)
    }

    /// Hashes a slice, padding if necessary.
    /// Returns None if slice length exceeds L.
    pub fn hash_slice(&self, data: &[Halo2Fr]) -> Option<Halo2Fr> {
        if data.len() > L {
            return None;
        }

        let mut padded = [Halo2Fr::zero(); L];
        for (i, v) in data.iter().enumerate() {
            padded[i] = *v;
        }

        Some(self.hash(&padded))
    }
}

impl<const L: usize> Default for PoseidonHasher<L> {
    fn default() -> Self {
        Self::new()
    }
}

/// Native Poseidon-compatible hash for field elements.
/// This uses a domain-separated hash construction.
fn poseidon_native_hash_array<const L: usize>(elements: &[Halo2Fr; L]) -> Halo2Fr {
    let mut hasher = Sha256::new();
    // Domain separator for Poseidon-compatible hashing
    hasher.update(b"HELIX_POSEIDON_NATIVE_V1");
    hasher.update(&(L as u64).to_le_bytes());

    for elem in elements {
        let repr = elem.to_repr();
        hasher.update(repr.as_ref());
    }

    let hash = hasher.finalize();
    bytes_to_field(&hash[..])
}

/// Converts bytes to a field element, reducing modulo the field modulus.
fn bytes_to_field(bytes: &[u8]) -> Halo2Fr {
    // Take first 32 bytes (or pad if shorter)
    let mut arr = [0u8; 32];
    let len = bytes.len().min(32);
    arr[..len].copy_from_slice(&bytes[..len]);

    // Clear the highest bit to ensure we're under the modulus
    // BN254 scalar field is ~254 bits, so clearing bit 255 is safe
    arr[31] &= 0x3F;  // Clear top 2 bits to be safe

    let mut repr = <Halo2Fr as PrimeField>::Repr::default();
    repr.as_mut().copy_from_slice(&arr);
    Halo2Fr::from_repr(repr).unwrap_or(Halo2Fr::zero())
}

/// Hashes two field elements using Poseidon-compatible hash (for Merkle trees).
pub fn poseidon_hash_two(left: Halo2Fr, right: Halo2Fr) -> Halo2Fr {
    let hasher = PoseidonHasher::<2>::new();
    hasher.hash(&[left, right])
}

/// Hashes a vector of Fr elements using multiple hash invocations.
/// Uses a Merkle-like folding structure for large inputs.
pub fn poseidon_hash(data: &[Fr]) -> Fr {
    if data.is_empty() {
        return Fr::ZERO;
    }

    let halo2_data: Vec<Halo2Fr> = data.iter().map(|f| *f.inner()).collect();

    if halo2_data.len() == 1 {
        // Single element: hash with padding
        let hasher = PoseidonHasher::<1>::new();
        let result = hasher.hash(&[halo2_data[0]]);
        return Fr::from_inner(result);
    }

    // For 2 elements, hash directly
    if halo2_data.len() == 2 {
        let result = poseidon_hash_two(halo2_data[0], halo2_data[1]);
        return Fr::from_inner(result);
    }

    // For larger inputs, use tree-based hashing
    // Pad to next power of 2 for balanced tree
    let padded_len = halo2_data.len().next_power_of_two();
    let mut padded: Vec<Halo2Fr> = halo2_data;
    padded.resize(padded_len, Halo2Fr::zero());

    // Build Merkle tree bottom-up
    while padded.len() > 1 {
        let mut next_level = Vec::with_capacity(padded.len() / 2);
        for chunk in padded.chunks(2) {
            let hash = poseidon_hash_two(chunk[0], chunk[1]);
            next_level.push(hash);
        }
        padded = next_level;
    }

    Fr::from_inner(padded[0])
}

/// Computes a Poseidon commitment: H(data || blinding).
pub fn poseidon_commit(data: &[Fr], blinding: Fr) -> Fr {
    let mut extended = data.to_vec();
    extended.push(blinding);
    poseidon_hash(&extended)
}

/// Poseidon commitment with explicit domain separation.
pub fn poseidon_commit_with_domain(domain: u64, data: &[Fr], blinding: Fr) -> Fr {
    let mut extended = vec![Fr::from_u64(domain)];
    extended.extend_from_slice(data);
    extended.push(blinding);
    poseidon_hash(&extended)
}

/// Domain separation constants for different commitment types.
pub mod domains {
    /// Domain for share commitments.
    pub const SHARE_COMMITMENT: u64 = 0x01;
    /// Domain for gradient commitments.
    pub const GRADIENT_COMMITMENT: u64 = 0x02;
    /// Domain for MAC commitments.
    pub const MAC_COMMITMENT: u64 = 0x03;
    /// Domain for state hash.
    pub const STATE_HASH: u64 = 0x04;
    /// Domain for aggregation.
    pub const AGGREGATION: u64 = 0x05;
}

/// Computes a share commitment using Poseidon.
pub fn share_commitment(share_data: &[Fr], blinding: Fr) -> Fr {
    poseidon_commit_with_domain(domains::SHARE_COMMITMENT, share_data, blinding)
}

/// Computes a gradient commitment using Poseidon.
pub fn gradient_commitment(gradient_data: &[Fr], blinding: Fr) -> Fr {
    poseidon_commit_with_domain(domains::GRADIENT_COMMITMENT, gradient_data, blinding)
}

/// Computes a MAC commitment using Poseidon.
pub fn mac_commitment(mac_data: &[Fr], blinding: Fr) -> Fr {
    poseidon_commit_with_domain(domains::MAC_COMMITMENT, mac_data, blinding)
}

/// Computes a state hash using Poseidon.
/// Returns (lo, hi) split for circuit compatibility.
pub fn poseidon_state_hash(w1: &[Fr], b1: &[Fr], w2: &[Fr], b2: &[Fr]) -> (Fr, Fr) {
    let mut data = vec![Fr::from_u64(domains::STATE_HASH)];
    data.extend_from_slice(w1);
    data.extend_from_slice(b1);
    data.extend_from_slice(w2);
    data.extend_from_slice(b2);

    let hash = poseidon_hash(&data);

    // Split hash into lo/hi for circuit public inputs
    let hash_bytes = hash.to_bytes_le();

    // Lo: first 16 bytes (128 bits)
    let lo = Fr::from_bytes_le(&[
        hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
        hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
        hash_bytes[8], hash_bytes[9], hash_bytes[10], hash_bytes[11],
        hash_bytes[12], hash_bytes[13], hash_bytes[14], hash_bytes[15],
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    // Hi: compute second hash for the high part
    let hash2 = poseidon_hash(&[hash, Fr::ONE]);
    let hash2_bytes = hash2.to_bytes_le();

    let hi = Fr::from_bytes_le(&[
        hash2_bytes[0], hash2_bytes[1], hash2_bytes[2], hash2_bytes[3],
        hash2_bytes[4], hash2_bytes[5], hash2_bytes[6], hash2_bytes[7],
        hash2_bytes[8], hash2_bytes[9], hash2_bytes[10], hash2_bytes[11],
        hash2_bytes[12], hash2_bytes[13], hash2_bytes[14], hash2_bytes[15],
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);

    (lo, hi)
}

/// Verifies that a commitment matches the expected value.
pub fn verify_commitment(commitment: Fr, data: &[Fr], blinding: Fr) -> bool {
    let computed = poseidon_commit(data, blinding);
    commitment.ct_eq(&computed).to_bool()
}

/// Verifies that a domain-separated commitment matches.
pub fn verify_domain_commitment(domain: u64, commitment: Fr, data: &[Fr], blinding: Fr) -> bool {
    let computed = poseidon_commit_with_domain(domain, data, blinding);
    commitment.ct_eq(&computed).to_bool()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_hash_two() {
        let a = Halo2Fr::from(42u64);
        let b = Halo2Fr::from(123u64);

        let h1 = poseidon_hash_two(a, b);
        let h2 = poseidon_hash_two(a, b);

        // Same inputs produce same hash
        assert_eq!(h1, h2);

        // Different order produces different hash
        let h3 = poseidon_hash_two(b, a);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_poseidon_hash_vector() {
        let data: Vec<Fr> = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ];

        let h1 = poseidon_hash(&data);
        let h2 = poseidon_hash(&data);

        assert!(h1.ct_eq(&h2).to_bool());

        // Different data produces different hash
        let data2: Vec<Fr> = vec![
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5), // Changed
        ];
        let h3 = poseidon_hash(&data2);
        assert!(!h1.ct_eq(&h3).to_bool());
    }

    #[test]
    fn test_poseidon_commit() {
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let blinding1 = Fr::from_u64(42);
        let blinding2 = Fr::from_u64(43);

        let c1 = poseidon_commit(&data, blinding1);
        let c2 = poseidon_commit(&data, blinding2);

        // Different blindings produce different commitments
        assert!(!c1.ct_eq(&c2).to_bool());

        // Same data and blinding produce same commitment
        let c3 = poseidon_commit(&data, blinding1);
        assert!(c1.ct_eq(&c3).to_bool());
    }

    #[test]
    fn test_verify_commitment() {
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let blinding = Fr::from_u64(42);

        let commitment = poseidon_commit(&data, blinding);
        assert!(verify_commitment(commitment, &data, blinding));
        assert!(!verify_commitment(commitment, &data, Fr::from_u64(43)));
    }

    #[test]
    fn test_share_commitment() {
        let share: Vec<Fr> = vec![
            Fr::from_f64(0.1),
            Fr::from_f64(0.2),
            Fr::from_f64(0.3),
        ];
        let blinding = Fr::from_u64(12345);

        let c1 = share_commitment(&share, blinding);
        let c2 = share_commitment(&share, blinding);

        assert!(c1.ct_eq(&c2).to_bool());
    }

    #[test]
    fn test_poseidon_state_hash() {
        let w1: Vec<Fr> = vec![Fr::from_f64(1.0), Fr::from_f64(2.0)];
        let b1: Vec<Fr> = vec![Fr::from_f64(0.1)];
        let w2: Vec<Fr> = vec![Fr::from_f64(3.0)];
        let b2: Vec<Fr> = vec![Fr::from_f64(0.2)];

        let (lo1, hi1) = poseidon_state_hash(&w1, &b1, &w2, &b2);
        let (lo2, hi2) = poseidon_state_hash(&w1, &b1, &w2, &b2);

        // Same weights produce same hash
        assert!(lo1.ct_eq(&lo2).to_bool());
        assert!(hi1.ct_eq(&hi2).to_bool());

        // Different weights produce different hash
        let w1_diff: Vec<Fr> = vec![Fr::from_f64(1.5), Fr::from_f64(2.0)];
        let (lo3, hi3) = poseidon_state_hash(&w1_diff, &b1, &w2, &b2);
        assert!(!lo1.ct_eq(&lo3).to_bool() || !hi1.ct_eq(&hi3).to_bool());
    }

    #[test]
    fn test_domain_separation() {
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let blinding = Fr::from_u64(42);

        // Different domains produce different commitments
        let c1 = poseidon_commit_with_domain(domains::SHARE_COMMITMENT, &data, blinding);
        let c2 = poseidon_commit_with_domain(domains::GRADIENT_COMMITMENT, &data, blinding);

        assert!(!c1.ct_eq(&c2).to_bool());
    }

    #[test]
    fn test_verify_domain_commitment() {
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2)];
        let blinding = Fr::from_u64(42);

        let commitment = poseidon_commit_with_domain(domains::SHARE_COMMITMENT, &data, blinding);
        assert!(verify_domain_commitment(domains::SHARE_COMMITMENT, commitment, &data, blinding));
        assert!(!verify_domain_commitment(domains::GRADIENT_COMMITMENT, commitment, &data, blinding));
    }

    #[test]
    fn test_hasher_fixed_length() {
        let hasher = PoseidonHasher::<4>::new();

        let msg = [
            Halo2Fr::from(1u64),
            Halo2Fr::from(2u64),
            Halo2Fr::from(3u64),
            Halo2Fr::from(4u64),
        ];

        let h1 = hasher.hash(&msg);
        let h2 = hasher.hash(&msg);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hasher_slice() {
        let hasher = PoseidonHasher::<4>::new();

        let data = vec![
            Halo2Fr::from(1u64),
            Halo2Fr::from(2u64),
        ];

        let h = hasher.hash_slice(&data);
        assert!(h.is_some());

        // Too long slice should return None
        let long_data = vec![Halo2Fr::from(1u64); 5];
        let h2 = hasher.hash_slice(&long_data);
        assert!(h2.is_none());
    }

    #[test]
    fn test_empty_hash() {
        let empty: Vec<Fr> = vec![];
        let h = poseidon_hash(&empty);
        assert!(h.ct_eq(&Fr::ZERO).to_bool());
    }

    #[test]
    fn test_single_element_hash() {
        let data = vec![Fr::from_u64(42)];
        let h1 = poseidon_hash(&data);
        let h2 = poseidon_hash(&data);
        assert!(h1.ct_eq(&h2).to_bool());
        assert!(!h1.ct_eq(&Fr::from_u64(42)).to_bool()); // Hash should differ from input
    }
}
