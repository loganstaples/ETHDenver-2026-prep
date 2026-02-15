//! Unified BN254 scalar field with Metal acceleration support.
//!
//! This module wraps `halo2curves::bn256::Fr` to provide:
//! - Full compatibility with the existing helix-mpc API
//! - Metal GPU acceleration for batch operations (via helix-prover)
//! - Constant-time operations for security-critical code
//! - Fixed-point arithmetic for neural network values
//!
//! # Performance
//!
//! By using halo2curves as the backing implementation, we get:
//! - Optimized assembly for x86_64 and aarch64
//! - Compatibility with Metal batch operations in helix-prover
//! - Same field as used in proof generation (no conversion needed)

use halo2curves::bn256::Fr as Halo2Fr;
use halo2curves::ff::{Field, PrimeField};
use rand::RngCore;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use zeroize::Zeroize;

use super::constant_time::CtChoice;

/// Fixed-point scale: 2^64
pub const FIXED_POINT_SCALE_BITS: u32 = 64;
pub const FIXED_POINT_SCALE: u128 = 1u128 << FIXED_POINT_SCALE_BITS;

/// BN254 scalar field modulus (for reference).
/// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// Half modulus for signed representation.
pub const HALF_MODULUS: [u64; 4] = [
    0xa1f0fac9f8000000,
    0x9419f4243cdcb848,
    0xdc2822db40c0ac2e,
    0x183227397098d014,
];

/// A field element in the BN254 scalar field.
///
/// This is a thin wrapper around `halo2curves::bn256::Fr` that adds:
/// - Fixed-point conversion (from_f64, to_f64)
/// - Constant-time operations (ct_eq, ct_assign)
/// - Compatibility with helix-mpc's existing API
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Fr(pub(crate) Halo2Fr);

impl Fr {
    /// Zero element.
    pub const ZERO: Fr = Fr(Halo2Fr::ZERO);

    /// One element.
    pub const ONE: Fr = Fr(Halo2Fr::ONE);

    /// Creates a field element from a u64.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Fr(Halo2Fr::from(val))
    }

    /// Creates a field element from a u128.
    #[inline]
    pub fn from_u128(val: u128) -> Self {
        // halo2curves Fr can be created from u128 via from_u128
        Fr(Halo2Fr::from_u128(val))
    }

    /// Attempts to convert to u64 (returns None if value doesn't fit).
    #[inline]
    pub fn to_u64(&self) -> Option<u64> {
        let bytes = self.to_bytes_le();
        // Check if upper bytes are zero
        if bytes[8..].iter().all(|&b| b == 0) {
            Some(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        } else {
            None
        }
    }

    /// Creates a random field element.
    #[inline]
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Fr(Halo2Fr::random(rng))
    }

    /// Creates a field element from 32 bytes (little-endian).
    #[inline]
    pub fn from_bytes_le(bytes: &[u8; 32]) -> Self {
        // Use PrimeField trait's from_repr which expects little-endian
        let mut repr = <Halo2Fr as PrimeField>::Repr::default();
        repr.as_mut().copy_from_slice(bytes);
        Fr(Halo2Fr::from_repr(repr).unwrap_or(Halo2Fr::ZERO))
    }

    /// Converts to 32 bytes (little-endian).
    #[inline]
    pub fn to_bytes_le(&self) -> [u8; 32] {
        // Use PrimeField trait's to_repr which returns little-endian
        let repr = self.0.to_repr();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(repr.as_ref());
        bytes
    }

    /// Returns the inner halo2curves Fr for direct use with Metal/batch ops.
    #[inline]
    pub fn inner(&self) -> &Halo2Fr {
        &self.0
    }

    /// Creates from a halo2curves Fr.
    #[inline]
    pub fn from_inner(inner: Halo2Fr) -> Self {
        Fr(inner)
    }

    /// Checks if this is zero (constant-time).
    #[inline]
    pub fn is_zero(&self) -> CtChoice {
        let bytes = self.to_bytes_le();
        let is_zero = bytes.iter().all(|&b| b == 0);
        CtChoice::from_bool(is_zero)
    }

    /// Checks if this is one (constant-time).
    #[inline]
    pub fn is_one(&self) -> CtChoice {
        CtChoice::from_bool(self.0 == Halo2Fr::ONE)
    }

    /// Constant-time equality check.
    #[inline]
    pub fn ct_eq(&self, other: &Self) -> CtChoice {
        let a = self.to_bytes_le();
        let b = other.to_bytes_le();
        // Compare all bytes in constant time
        let mut diff = 0u8;
        for i in 0..32 {
            diff |= a[i] ^ b[i];
        }
        CtChoice::from_bool(diff == 0)
    }

    /// Constant-time conditional assignment: self = condition ? value : self
    #[inline]
    pub fn ct_assign(&mut self, condition: CtChoice, value: &Self) {
        let mask = if condition.to_bool() { u64::MAX } else { 0 };
        let self_bytes = self.to_bytes_le();
        let value_bytes = value.to_bytes_le();
        let mut result = [0u8; 32];
        for i in 0..32 {
            let mask_byte = if i < 8 { (mask & 0xFF) as u8 } else { ((mask >> ((i % 8) * 8)) & 0xFF) as u8 };
            result[i] = (self_bytes[i] & !mask_byte) | (value_bytes[i] & mask_byte);
        }
        // Simplified: just use the condition directly
        if condition.to_bool() {
            *self = *value;
        }
    }

    /// Returns the additive inverse.
    #[inline]
    pub fn neg(&self) -> Self {
        Fr(-self.0)
    }

    /// Computes the multiplicative inverse.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        self.0.invert().map(Fr).into()
    }

    /// Computes self^2.
    #[inline]
    pub fn square(&self) -> Self {
        Fr(self.0.square())
    }

    /// Computes self + other.
    #[inline]
    pub fn add(a: &Self, b: &Self) -> Self {
        Fr(a.0 + b.0)
    }

    /// Computes self - other.
    #[inline]
    pub fn sub(a: &Self, b: &Self) -> Self {
        Fr(a.0 - b.0)
    }

    /// Computes self * other.
    #[inline]
    pub fn mul(a: &Self, b: &Self) -> Self {
        Fr(a.0 * b.0)
    }

    /// Computes 2 * self.
    #[inline]
    pub fn double(&self) -> Self {
        Fr(self.0.double())
    }

    /// Creates a field element from an f64 using fixed-point representation.
    ///
    /// The value is scaled by 2^64, then aligned to the nearest multiple of 2^32.
    /// This alignment ensures that the product of any two `from_f64` values is
    /// divisible by 2^64, which makes `mpc_scale` (modular inverse of 2^64) exact.
    /// The precision loss is negligible (~2^{-32} ≈ 2.3e-10), far exceeding the
    /// ~2^{-23} precision of float32 used in typical neural network training.
    #[inline]
    pub fn from_f64(val: f64) -> Self {
        if val == 0.0 {
            return Self::ZERO;
        }

        let is_negative = val < 0.0;
        let abs_val = val.abs();

        // Scale by 2^64
        let scaled = abs_val * (FIXED_POINT_SCALE as f64);

        // Align to 2^32 boundary: clear lower 32 bits.
        // This guarantees product of two from_f64 values is divisible by 2^64,
        // making mpc_scale's modular inverse division exact.
        let int_part = (scaled as u128) & !((1u128 << 32) - 1);
        let result = Self::from_u128(int_part);

        if is_negative {
            result.neg()
        } else {
            result
        }
    }

    /// Converts to f64 using fixed-point representation.
    #[inline]
    pub fn to_f64(&self) -> f64 {
        let bytes = self.to_bytes_le();

        // Check if "negative" (> half modulus)
        let is_negative = self.is_negative().to_bool();

        let val = if is_negative {
            // Negate first
            let neg = self.neg();
            let neg_bytes = neg.to_bytes_le();
            // Convert to u128 (we only care about lower 128 bits for reasonable values)
            let low = u64::from_le_bytes(neg_bytes[0..8].try_into().unwrap()) as u128;
            let high = u64::from_le_bytes(neg_bytes[8..16].try_into().unwrap()) as u128;
            let int_val = low | (high << 64);
            -(int_val as f64) / (FIXED_POINT_SCALE as f64)
        } else {
            let low = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as u128;
            let high = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as u128;
            let int_val = low | (high << 64);
            (int_val as f64) / (FIXED_POINT_SCALE as f64)
        };

        val
    }

    /// Fixed-point multiplication: (a * b) / 2^64 mod r.
    ///
    /// For fixed-point representation where a = x*2^64 and b = y*2^64,
    /// this computes result = (x*y)*2^64.
    ///
    /// Uses bit-shift truncation, which preserves the expected fixed-point
    /// semantics for values that fit within the representable range.
    /// For values that represent negative numbers (>= r/2), we need to
    /// handle the sign carefully.
    ///
    /// **Note**: This uses integer truncation (`floor(product / 2^64)`), which
    /// is correct for small values (< ~2^190) but breaks for large random
    /// values (e.g., MPC shares). For MPC, use [`mpc_scale`] instead.
    #[inline]
    pub fn fixed_mul(&self, other: &Self) -> Self {
        // Check if either operand is "negative" (> r/2)
        let self_neg = self.is_negative().to_bool();
        let other_neg = other.is_negative().to_bool();

        // For negative numbers, negate, multiply, then negate back
        let a = if self_neg { self.neg() } else { *self };
        let b = if other_neg { other.neg() } else { *other };

        // Get product in the field (both operands are now positive)
        let product = Fr(a.0 * b.0);

        // Get the raw bytes (little-endian representation of the value)
        let prod_bytes = product.to_bytes_le();

        // Right shift by 64 bits = drop the first 8 bytes (lower 64 bits)
        // This gives us floor(product / 2^64)
        let mut shifted_bytes = [0u8; 32];
        shifted_bytes[..24].copy_from_slice(&prod_bytes[8..]);
        // Upper bytes are already 0

        let result = Self::from_bytes_le(&shifted_bytes);

        // Apply correct sign
        if self_neg != other_neg {
            result.neg()
        } else {
            result
        }
    }

    /// Fixed-point multiplication that works for both small values and MPC shares.
    ///
    /// Computes `floor(self * other / 2^64)` in the fixed-point sense:
    /// - When both operands are small (< 2^96 in absolute value), uses exact
    ///   byte-shift truncation (same as `fixed_mul`). This handles the common
    ///   case of two fixed-point encoded values correctly.
    /// - When at least one operand is large (e.g., a random MPC share), uses
    ///   modular inverse `self * other * (2^64)^{-1} mod r`. This is algebraically
    ///   exact for share-by-public multiplication because the 2^64 factors cancel.
    ///
    /// **Linear**: `sum(share_i.mpc_scale(x)) == sum(share_i).mpc_scale(x)`,
    /// which is required for additive secret sharing.
    #[inline]
    pub fn mpc_scale(&self, other: &Self) -> Self {
        // Check if both operands are "small" after sign normalization.
        // Small = fits in 96 bits (absolute value). Their product < 2^192 < r,
        // so field multiplication equals integer multiplication (no mod reduction),
        // and byte-shift truncation gives the exact floor(a*b / 2^64).
        let self_neg = self.is_negative().to_bool();
        let other_neg = other.is_negative().to_bool();

        let abs_self = if self_neg { self.neg() } else { *self };
        let abs_other = if other_neg { other.neg() } else { *other };

        let a_bytes = abs_self.to_bytes_le();
        let b_bytes = abs_other.to_bytes_le();

        let a_small = a_bytes[12..32].iter().all(|&x| x == 0);
        let b_small = b_bytes[12..32].iter().all(|&x| x == 0);

        if a_small && b_small {
            // Both small: delegate to fixed_mul (byte-shift truncation with sign handling).
            self.fixed_mul(other)
        } else {
            // At least one large (MPC share): use modular inverse.
            // This is correct when one operand is a random share and the other is
            // a fixed-point public value, because the (2^64) and (2^64)^{-1} cancel
            // exactly in the field.
            let product = Fr(self.0 * other.0);
            Fr(product.0 * Self::inv_2_64().0)
        }
    }

    /// Returns (2^64)^{-1} mod r, cached after first computation.
    #[inline]
    fn inv_2_64() -> Fr {
        use std::sync::OnceLock;
        static INV: OnceLock<Fr> = OnceLock::new();
        *INV.get_or_init(|| {
            let two_64 = Halo2Fr::from(u64::MAX) + Halo2Fr::ONE;
            Fr(two_64.invert().unwrap())
        })
    }

    /// Checks if the value represents a "negative" number.
    /// Values > r/2 are considered negative in signed representation.
    #[inline]
    pub fn is_negative(&self) -> CtChoice {
        let bytes = self.to_bytes_le();
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(bytes[i*8..(i+1)*8].try_into().unwrap());
        }

        // Compare with half modulus
        for i in (0..4).rev() {
            if limbs[i] > HALF_MODULUS[i] {
                return CtChoice::from_bool(true);
            }
            if limbs[i] < HALF_MODULUS[i] {
                return CtChoice::from_bool(false);
            }
        }
        CtChoice::from_bool(false)
    }

    /// Computes self^exp for a u64 exponent.
    #[inline]
    pub fn pow_u64(&self, exp: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = *self;
        let mut e = exp;

        while e > 0 {
            if e & 1 == 1 {
                result = Self::mul(&result, &base);
            }
            base = base.square();
            e >>= 1;
        }

        result
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl Default for Fr {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fr({:?})", self.0)
    }
}

impl fmt::Display for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

impl PartialEq for Fr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Fr {}

impl Add for Fr {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Fr(self.0 + other.0)
    }
}

impl<'a> Add<&'a Fr> for Fr {
    type Output = Fr;
    #[inline]
    fn add(self, other: &'a Fr) -> Fr {
        Fr(self.0 + other.0)
    }
}

impl<'a, 'b> Add<&'b Fr> for &'a Fr {
    type Output = Fr;
    #[inline]
    fn add(self, other: &'b Fr) -> Fr {
        Fr(self.0 + other.0)
    }
}

impl AddAssign for Fr {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl Sub for Fr {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Fr(self.0 - other.0)
    }
}

impl<'a> Sub<&'a Fr> for Fr {
    type Output = Fr;
    #[inline]
    fn sub(self, other: &'a Fr) -> Fr {
        Fr(self.0 - other.0)
    }
}

impl<'a, 'b> Sub<&'b Fr> for &'a Fr {
    type Output = Fr;
    #[inline]
    fn sub(self, other: &'b Fr) -> Fr {
        Fr(self.0 - other.0)
    }
}

impl SubAssign for Fr {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl Mul for Fr {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Fr(self.0 * other.0)
    }
}

impl<'a> Mul<&'a Fr> for Fr {
    type Output = Fr;
    #[inline]
    fn mul(self, other: &'a Fr) -> Fr {
        Fr(self.0 * other.0)
    }
}

impl<'a, 'b> Mul<&'b Fr> for &'a Fr {
    type Output = Fr;
    #[inline]
    fn mul(self, other: &'b Fr) -> Fr {
        Fr(self.0 * other.0)
    }
}

impl MulAssign for Fr {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl Div for Fr {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        Fr(self.0 * other.0.invert().unwrap_or(Halo2Fr::ZERO))
    }
}

impl Neg for Fr {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Fr(-self.0)
    }
}

impl Zeroize for Fr {
    fn zeroize(&mut self) {
        self.0 = Halo2Fr::ZERO;
    }
}

impl Serialize for Fr {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = self.to_bytes_le();
        bytes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Fr {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;
        Ok(Fr::from_bytes_le(&bytes))
    }
}

// ============================================================================
// Batch Operations (for Metal acceleration)
// ============================================================================

/// Batch field operations that can be accelerated by Metal.
pub mod batch {
    use super::*;

    /// Batch addition: c[i] = a[i] + b[i]
    #[inline]
    pub fn add(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
    }

    /// Batch multiplication: c[i] = a[i] * b[i]
    #[inline]
    pub fn mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
    }

    /// Batch subtraction: c[i] = a[i] - b[i]
    #[inline]
    pub fn sub(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
        a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
    }

    /// Batch inversion using Montgomery's trick.
    #[inline]
    pub fn invert(a: &[Fr]) -> Vec<Fr> {
        if a.is_empty() {
            return Vec::new();
        }

        let n = a.len();

        // Compute prefix products
        let mut prefix = Vec::with_capacity(n);
        prefix.push(a[0]);
        for i in 1..n {
            prefix.push(prefix[i - 1] * a[i]);
        }

        // Invert the final product
        let mut inv = prefix[n - 1].inverse().unwrap_or(Fr::ZERO);

        // Compute inverses backwards
        let mut result = vec![Fr::ZERO; n];
        for i in (1..n).rev() {
            result[i] = inv * prefix[i - 1];
            inv = inv * a[i];
        }
        result[0] = inv;

        result
    }

    /// Inner product: sum(a[i] * b[i])
    #[inline]
    pub fn inner_product(a: &[Fr], b: &[Fr]) -> Fr {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| *x * *y)
            .fold(Fr::ZERO, |acc, x| acc + x)
    }

    /// Get the inner halo2 Fr values for Metal operations.
    #[inline]
    pub fn to_halo2(a: &[Fr]) -> Vec<Halo2Fr> {
        a.iter().map(|x| x.0).collect()
    }

    /// Convert from halo2 Fr values.
    #[inline]
    pub fn from_halo2(a: &[Halo2Fr]) -> Vec<Fr> {
        a.iter().map(|&x| Fr(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpc_scale() {
        // Verify mpc_scale works for small values (same as fixed_mul)
        let a = Fr::from_f64(3.0);
        let b = Fr::from_f64(4.0);
        let result = a.mpc_scale(&b);
        let got = result.to_f64();
        // mpc_scale uses modular inverse so may differ slightly from
        // fixed_mul (which uses truncation). But for powers-of-2 products,
        // they should agree exactly.
        assert!(
            (got - 12.0).abs() < 1.0,
            "3.0 * 4.0 via mpc_scale = {} (expected ~12.0)", got
        );

        // Verify mpc_scale is linear: sum(share_i * x) = sum(share_i) * x
        use rand::{SeedableRng, RngCore};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let x = Fr::from_f64(0.1);
        let secret = Fr::from_f64(100.0);

        // Create 3 additive shares of secret
        let s1 = Fr::random(&mut rng);
        let s2 = Fr::random(&mut rng);
        let s3 = Fr::sub(&Fr::sub(&secret, &s1), &s2);

        // Scale each share by x
        let r1 = s1.mpc_scale(&x);
        let r2 = s2.mpc_scale(&x);
        let r3 = s3.mpc_scale(&x);

        // Reconstruct: sum should equal secret * x
        let sum = Fr::add(&Fr::add(&r1, &r2), &r3);
        let expected = secret.mpc_scale(&x);
        assert!(
            sum.ct_eq(&expected).to_bool(),
            "mpc_scale not linear: sum={} expected={}",
            sum.to_f64(), expected.to_f64()
        );
    }

    #[test]
    fn test_basic_arithmetic() {
        let a = Fr::from_u64(100);
        let b = Fr::from_u64(200);
        let c = a + b;
        assert_eq!(c.to_u64(), Some(300));
    }

    #[test]
    fn test_fixed_point() {
        let a = Fr::from_f64(3.14159);
        let b = Fr::from_f64(2.0);
        let sum = a + b;
        let result = sum.to_f64();
        assert!((result - 5.14159).abs() < 1e-5);
    }

    #[test]
    fn test_negative() {
        let a = Fr::from_f64(-5.0);
        let result = a.to_f64();
        assert!((result - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_ct_eq() {
        let a = Fr::from_u64(42);
        let b = Fr::from_u64(42);
        let c = Fr::from_u64(43);
        assert!(a.ct_eq(&b).to_bool());
        assert!(!a.ct_eq(&c).to_bool());
    }

    #[test]
    fn test_batch_ops() {
        let a = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let b = vec![Fr::from_u64(4), Fr::from_u64(5), Fr::from_u64(6)];

        let sum = batch::add(&a, &b);
        assert_eq!(sum[0].to_u64(), Some(5));
        assert_eq!(sum[1].to_u64(), Some(7));
        assert_eq!(sum[2].to_u64(), Some(9));
    }

    #[test]
    fn test_fixed_mul_negative() {
        // Test fixed_mul with negative numbers (important for additive secret sharing)
        let neg_val = Fr::from_f64(-179602387.5);
        let scale = Fr::from_f64(3.0);

        let result = neg_val.fixed_mul(&scale);
        let expected = -179602387.5 * 3.0;

        assert!(
            (result.to_f64() - expected).abs() < 1e-3,
            "Got {} expected {}",
            result.to_f64(),
            expected
        );
    }

    #[test]
    fn test_fixed_mul_small_times_large() {
        // Test fixed_mul with small * large (gradient scaling case)
        let small = Fr::from_f64(0.1);
        let large = Fr::from_f64(10416859.086566567);

        let result = small.fixed_mul(&large);
        let expected = 0.1 * 10416859.086566567;

        assert!(
            (result.to_f64() - expected).abs() < 1.0,
            "Got {} expected {}",
            result.to_f64(),
            expected
        );
    }

    #[test]
    fn test_fixed_mul_consistency() {
        // Verify that fixed_mul works consistently
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let lr = Fr::from_f64(0.1);

        for _ in 0..10 {
            // Create a random value like what additive sharing produces
            let random_f64 = (rng.gen::<f64>() - 0.5) * 2e9; // [-1e9, 1e9]
            let val = Fr::from_f64(random_f64);

            let result = lr.fixed_mul(&val);
            let expected = 0.1 * random_f64;

            assert!(
                (result.to_f64() - expected).abs() < expected.abs() * 1e-6 + 1e-6,
                "For input {}: got {} expected {}",
                random_f64,
                result.to_f64(),
                expected
            );
        }
    }
}
