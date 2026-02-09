//! Constant-time operations for cryptographic security.
//!
//! These operations ensure that execution time is independent of the values
//! being processed, preventing timing side-channel attacks. This is critical
//! for MPC where secret shares must not leak information through timing.
//!
//! # Security Model
//!
//! Timing attacks can leak secrets by observing how long operations take.
//! For example, early-exit comparisons or data-dependent branches can reveal
//! information about secret values. These operations use bitwise arithmetic
//! to ensure constant execution time.

use zeroize::Zeroize;

/// A constant-time boolean choice.
///
/// This type ensures that boolean decisions don't leak through timing.
/// Internally, it's stored as 0 or u64::MAX to enable constant-time selection.
#[derive(Debug, Clone, Copy)]
pub struct CtChoice(u64);

impl CtChoice {
    /// Creates a choice representing `false` (0).
    #[inline]
    pub const fn false_value() -> Self {
        CtChoice(0)
    }

    /// Creates a choice representing `true` (all 1s).
    #[inline]
    pub const fn true_value() -> Self {
        CtChoice(u64::MAX)
    }

    /// Creates a choice from a boolean (constant-time conversion).
    #[inline]
    pub fn from_bool(b: bool) -> Self {
        // Convert bool to 0 or 1, then to 0 or MAX
        let val = b as u64;
        // Negate and wrap: 0 -> 0, 1 -> MAX
        CtChoice(val.wrapping_neg())
    }

    /// Creates a choice from whether a value is zero (constant-time).
    #[inline]
    pub fn is_zero_u64(val: u64) -> Self {
        // If val is 0, result is true (MAX)
        // Uses the fact that (val | -val) >> 63 is 1 if val != 0, 0 if val == 0
        let is_nonzero = ((val | val.wrapping_neg()) >> 63) as u64;
        // Flip: if nonzero, return 0; if zero, return MAX
        CtChoice(is_nonzero.wrapping_sub(1))
    }

    /// Creates a choice from whether a value is nonzero (constant-time).
    #[inline]
    pub fn is_nonzero_u64(val: u64) -> Self {
        Self::is_zero_u64(val).not_ct()
    }

    /// Constant-time NOT operation.
    #[inline]
    pub fn not_ct(self) -> Self {
        CtChoice(!self.0)
    }

    /// Constant-time AND operation.
    #[inline]
    pub fn and_ct(self, other: Self) -> Self {
        CtChoice(self.0 & other.0)
    }

    /// Constant-time OR operation.
    #[inline]
    pub fn or_ct(self, other: Self) -> Self {
        CtChoice(self.0 | other.0)
    }

    /// Constant-time XOR operation.
    #[inline]
    pub fn xor_ct(self, other: Self) -> Self {
        CtChoice(self.0 ^ other.0)
    }

    /// Returns the underlying mask value.
    #[inline]
    pub fn mask(self) -> u64 {
        self.0
    }

    /// Converts to bool (use with caution - may introduce timing variations).
    #[inline]
    pub fn to_bool(self) -> bool {
        self.0 != 0
    }

    /// Constant-time select: returns `a` if choice is true, `b` otherwise.
    #[inline]
    pub fn select_u64(self, a: u64, b: u64) -> u64 {
        // If mask is all 1s (true), return a
        // If mask is all 0s (false), return b
        // result = (mask & a) | (!mask & b)
        (self.0 & a) | (!self.0 & b)
    }

    /// Constant-time select for arrays.
    #[inline]
    pub fn select_array<const N: usize>(self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut result = [0u64; N];
        for i in 0..N {
            result[i] = self.select_u64(a[i], b[i]);
        }
        result
    }
}

/// Constant-time comparison: returns true if a == b.
#[inline]
pub fn ct_eq_u64(a: u64, b: u64) -> CtChoice {
    // XOR gives 0 if equal, nonzero otherwise
    CtChoice::is_zero_u64(a ^ b)
}

/// Constant-time comparison: returns true if a < b (unsigned).
#[inline]
pub fn ct_lt_u64(a: u64, b: u64) -> CtChoice {
    // (a - b) will have high bit set if a < b (borrow occurred)
    // We need to handle overflow correctly
    let diff = a.wrapping_sub(b);
    let borrow = (((!a) & b) | (((!a) | b) & diff)) >> 63;
    CtChoice::from_bool(borrow != 0)
}

/// Constant-time comparison: returns true if a > b (unsigned).
#[inline]
pub fn ct_gt_u64(a: u64, b: u64) -> CtChoice {
    ct_lt_u64(b, a)
}

/// Constant-time comparison: returns true if a <= b (unsigned).
#[inline]
pub fn ct_le_u64(a: u64, b: u64) -> CtChoice {
    ct_gt_u64(a, b).not_ct()
}

/// Constant-time comparison: returns true if a >= b (unsigned).
#[inline]
pub fn ct_ge_u64(a: u64, b: u64) -> CtChoice {
    ct_lt_u64(a, b).not_ct()
}

/// Constant-time comparison for arrays (lexicographic, big-endian).
/// Returns true if a < b.
#[inline]
pub fn ct_lt_array<const N: usize>(a: &[u64; N], b: &[u64; N]) -> CtChoice {
    // Process from most significant limb to least
    let mut result = CtChoice::false_value();
    let mut equal_so_far = CtChoice::true_value();

    for i in (0..N).rev() {
        let lt_here = ct_lt_u64(a[i], b[i]);
        let gt_here = ct_gt_u64(a[i], b[i]);

        // If we're still equal and a[i] < b[i], then a < b
        result = result.or_ct(equal_so_far.and_ct(lt_here));

        // Update equal_so_far: stays true only if current limbs are equal
        equal_so_far = equal_so_far.and_ct(lt_here.not_ct()).and_ct(gt_here.not_ct());
    }

    result
}

/// Constant-time comparison for arrays: returns true if a == b.
#[inline]
pub fn ct_eq_array<const N: usize>(a: &[u64; N], b: &[u64; N]) -> CtChoice {
    let mut xor_acc = 0u64;
    for i in 0..N {
        xor_acc |= a[i] ^ b[i];
    }
    CtChoice::is_zero_u64(xor_acc)
}

/// Constant-time comparison for arrays: returns true if a >= b.
#[inline]
pub fn ct_ge_array<const N: usize>(a: &[u64; N], b: &[u64; N]) -> CtChoice {
    ct_lt_array(a, b).not_ct()
}

/// Constant-time conditional swap: swaps a and b if condition is true.
#[inline]
pub fn ct_swap_u64(condition: CtChoice, a: &mut u64, b: &mut u64) {
    let mask = condition.mask();
    let diff = (*a ^ *b) & mask;
    *a ^= diff;
    *b ^= diff;
}

/// Constant-time conditional swap for arrays.
#[inline]
pub fn ct_swap_array<const N: usize>(condition: CtChoice, a: &mut [u64; N], b: &mut [u64; N]) {
    let mask = condition.mask();
    for i in 0..N {
        let diff = (a[i] ^ b[i]) & mask;
        a[i] ^= diff;
        b[i] ^= diff;
    }
}

/// Constant-time conditional assign: sets target = value if condition is true.
#[inline]
pub fn ct_assign_u64(condition: CtChoice, target: &mut u64, value: u64) {
    *target = condition.select_u64(value, *target);
}

/// Constant-time conditional assign for arrays.
#[inline]
pub fn ct_assign_array<const N: usize>(condition: CtChoice, target: &mut [u64; N], value: &[u64; N]) {
    for i in 0..N {
        target[i] = condition.select_u64(value[i], target[i]);
    }
}

/// Constant-time comparison for 32-byte hash values.
///
/// Compares byte-by-byte using XOR accumulation to avoid early-exit
/// timing leaks that could reveal information about secret commitments.
#[inline]
pub fn ct_eq_hash(a: &[u8; 32], b: &[u8; 32]) -> CtChoice {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    // diff == 0 iff a == b
    CtChoice::from_bool(diff == 0)
}

/// A secure buffer that zeroizes on drop.
#[derive(Clone)]
pub struct SecureBuffer<const N: usize> {
    data: [u64; N],
}

impl<const N: usize> SecureBuffer<N> {
    /// Creates a new zeroed buffer.
    pub fn new() -> Self {
        Self { data: [0u64; N] }
    }

    /// Creates a buffer from an array.
    pub fn from_array(data: [u64; N]) -> Self {
        Self { data }
    }

    /// Returns a reference to the data.
    pub fn as_array(&self) -> &[u64; N] {
        &self.data
    }

    /// Returns a mutable reference to the data.
    pub fn as_array_mut(&mut self) -> &mut [u64; N] {
        &mut self.data
    }

    /// Constant-time equality check.
    pub fn ct_eq(&self, other: &Self) -> CtChoice {
        ct_eq_array(&self.data, &other.data)
    }

    /// Constant-time conditional assign.
    pub fn ct_assign(&mut self, condition: CtChoice, value: &Self) {
        ct_assign_array(condition, &mut self.data, &value.data);
    }
}

impl<const N: usize> Default for SecureBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Drop for SecureBuffer<N> {
    fn drop(&mut self) {
        self.data.zeroize();
    }
}

impl<const N: usize> Zeroize for SecureBuffer<N> {
    fn zeroize(&mut self) {
        self.data.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_choice_from_bool() {
        assert_eq!(CtChoice::from_bool(true).mask(), u64::MAX);
        assert_eq!(CtChoice::from_bool(false).mask(), 0);
    }

    #[test]
    fn test_ct_choice_is_zero() {
        assert!(CtChoice::is_zero_u64(0).to_bool());
        assert!(!CtChoice::is_zero_u64(1).to_bool());
        assert!(!CtChoice::is_zero_u64(u64::MAX).to_bool());
    }

    #[test]
    fn test_ct_choice_is_nonzero() {
        assert!(!CtChoice::is_nonzero_u64(0).to_bool());
        assert!(CtChoice::is_nonzero_u64(1).to_bool());
        assert!(CtChoice::is_nonzero_u64(u64::MAX).to_bool());
    }

    #[test]
    fn test_ct_choice_select() {
        let t = CtChoice::true_value();
        let f = CtChoice::false_value();

        assert_eq!(t.select_u64(10, 20), 10);
        assert_eq!(f.select_u64(10, 20), 20);
    }

    #[test]
    fn test_ct_eq_u64() {
        assert!(ct_eq_u64(5, 5).to_bool());
        assert!(!ct_eq_u64(5, 6).to_bool());
        assert!(ct_eq_u64(0, 0).to_bool());
        assert!(ct_eq_u64(u64::MAX, u64::MAX).to_bool());
    }

    #[test]
    fn test_ct_lt_u64() {
        assert!(ct_lt_u64(3, 5).to_bool());
        assert!(!ct_lt_u64(5, 3).to_bool());
        assert!(!ct_lt_u64(5, 5).to_bool());
        assert!(ct_lt_u64(0, 1).to_bool());
        assert!(ct_lt_u64(0, u64::MAX).to_bool());
    }

    #[test]
    fn test_ct_gt_u64() {
        assert!(ct_gt_u64(5, 3).to_bool());
        assert!(!ct_gt_u64(3, 5).to_bool());
        assert!(!ct_gt_u64(5, 5).to_bool());
    }

    #[test]
    fn test_ct_swap() {
        let mut a = 10u64;
        let mut b = 20u64;

        ct_swap_u64(CtChoice::true_value(), &mut a, &mut b);
        assert_eq!(a, 20);
        assert_eq!(b, 10);

        ct_swap_u64(CtChoice::false_value(), &mut a, &mut b);
        assert_eq!(a, 20);
        assert_eq!(b, 10);
    }

    #[test]
    fn test_ct_eq_array() {
        let a = [1u64, 2, 3, 4];
        let b = [1u64, 2, 3, 4];
        let c = [1u64, 2, 3, 5];

        assert!(ct_eq_array(&a, &b).to_bool());
        assert!(!ct_eq_array(&a, &c).to_bool());
    }

    #[test]
    fn test_ct_lt_array() {
        // Lexicographic comparison (big-endian: last element is most significant)
        let a = [1u64, 2, 3, 4];
        let b = [1u64, 2, 3, 5];
        let c = [1u64, 2, 4, 4];

        assert!(ct_lt_array(&a, &b).to_bool()); // 4 < 5 at position 3
        assert!(ct_lt_array(&a, &c).to_bool()); // 3 < 4 at position 2
        assert!(!ct_lt_array(&b, &a).to_bool());
        assert!(!ct_lt_array(&a, &a).to_bool()); // equal
    }

    #[test]
    fn test_secure_buffer_zeroize() {
        let mut buf = SecureBuffer::<4>::from_array([1, 2, 3, 4]);
        buf.zeroize();
        assert_eq!(buf.as_array(), &[0, 0, 0, 0]);
    }
}
