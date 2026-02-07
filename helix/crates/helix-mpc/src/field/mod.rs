//! Finite field arithmetic for secure MPC.
//!
//! This module provides cryptographically secure field arithmetic using the
//! BN254 scalar field. All operations are constant-time to prevent timing
//! side-channel attacks.
//!
//! # Why Field Arithmetic?
//!
//! MPC protocols require arithmetic over finite fields rather than floating-point:
//!
//! 1. **Exact arithmetic**: Field operations are deterministic and exact,
//!    unlike floating-point which has rounding errors.
//!
//! 2. **Cryptographic security**: Field elements can be secret-shared such that
//!    any subset of shares reveals nothing about the secret.
//!
//! 3. **ZK compatibility**: The BN254 field is used in Ethereum's ZK precompiles,
//!    allowing proofs to be verified on-chain.
//!
//! # Fixed-Point Representation
//!
//! Neural network values (weights, activations, gradients) are represented as
//! fixed-point numbers. A value x is stored as `x * 2^64 mod r`, allowing
//! roughly 64 bits of precision for the fractional part.
//!
//! # Constant-Time Operations
//!
//! All field operations are implemented to run in constant time regardless of
//! the values being processed. This prevents timing attacks that could leak
//! information about secret shares.
//!
//! # Metal/GPU Acceleration
//!
//! The field implementation uses `halo2curves::bn256::Fr` internally, which
//! allows compatibility with Metal-accelerated batch operations in helix-prover.
//! The `batch` submodule provides operations that can be accelerated.
//!
//! # Example
//!
//! ```rust
//! use helix_mpc::field::{Fr, ops};
//!
//! // Create field elements from f64 (for compatibility)
//! let a = Fr::from_f64(3.5);
//! let b = Fr::from_f64(2.0);
//!
//! // Field arithmetic
//! let sum = a + b;
//! let product = a.fixed_mul(&b);
//!
//! // Convert back (for display/debugging)
//! println!("sum = {}", sum.to_f64());
//! println!("product = {}", product.to_f64());
//! ```

// Legacy implementation kept for reference/fallback
mod bn254;
pub mod constant_time;
pub mod ops;
pub mod unified;

// Re-export main types from unified (halo2curves-backed) implementation
pub use unified::Fr;
pub use unified::batch;
pub use unified::{FIXED_POINT_SCALE, FIXED_POINT_SCALE_BITS, HALF_MODULUS, MODULUS};
pub use constant_time::{
    ct_assign_array, ct_assign_u64, ct_eq_array, ct_eq_hash, ct_eq_u64, ct_ge_array, ct_ge_u64,
    ct_gt_u64, ct_le_u64, ct_lt_array, ct_lt_u64, ct_swap_array, ct_swap_u64, CtChoice,
    SecureBuffer,
};
pub use ops::{
    add_vec, from_f64_vec, from_u64_vec, inner_product, inner_product_fixed, lagrange_interpolate,
    matmul, matmul_fixed, matvec, mul_vec, mul_vec_fixed, neg_vec, outer_product, poly_eval,
    random_matrix, random_vec, scale_vec, sub_vec, sum, to_f64_vec, transpose, zeroize_vec,
    SecureVec,
};

/// Type alias for field element (for clarity in MPC code).
pub type FieldElement = Fr;

/// Creates a field element from a u64.
#[inline]
pub fn fr(val: u64) -> Fr {
    Fr::from_u64(val)
}

/// Creates a field element from an f64 (fixed-point).
#[inline]
pub fn fr_f64(val: f64) -> Fr {
    Fr::from_f64(val)
}

/// Zero field element.
#[inline]
pub const fn zero() -> Fr {
    Fr::ZERO
}

/// One field element.
#[inline]
pub const fn one() -> Fr {
    Fr::ONE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        let a = fr(100);
        let b = fr(200);
        let c = a + b;
        assert_eq!(c.to_u64(), Some(300));
    }

    #[test]
    fn test_fixed_point_exports() {
        let a = fr_f64(3.14159);
        let b = fr_f64(2.71828);
        let c = a + b;
        let result = c.to_f64();
        assert!((result - 5.85987).abs() < 1e-5);
    }

    #[test]
    fn test_zero_one() {
        assert!(zero().is_zero().to_bool());
        assert!(one().is_one().to_bool());
    }
}
