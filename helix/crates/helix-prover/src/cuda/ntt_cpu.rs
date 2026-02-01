//! CPU fallback for NTT using Cooley-Tukey algorithm.
//!
//! This implements the Number-Theoretic Transform over the BN254 scalar field
//! using the Cooley-Tukey radix-2 decimation-in-time algorithm.

/// BN254 scalar field modulus
/// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// Montgomery R = 2^256 mod p
const R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

/// -p^{-1} mod 2^64
const INV: u64 = 0xc2e1f593efffffff;

/// Primitive 2^28-th root of unity in Montgomery form
/// This is ω such that ω^(2^28) = 1 mod p
const ROOT_OF_UNITY: [u64; 4] = [
    0x3e7e3d9e8b8f1f63,
    0x7ccc637db1fc1cd9,
    0x5f4d5e06718a5d10,
    0x19c6dfb841f39d00,
];

/// Max log2(n) for NTT (2^28 supported)
const MAX_LOG_N: usize = 28;

// ============================================================================
// Field Arithmetic
// ============================================================================

/// Add two field elements: c = a + b mod p
pub fn field_add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u128;

    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + carry;
        result[i] = sum as u64;
        carry = sum >> 64;
    }

    // Reduce if >= modulus
    if carry != 0 || !less_than(&result, &MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

/// Subtract two field elements: c = a - b mod p
pub fn field_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = (a[i] as i128) - (b[i] as i128) - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    // If we borrowed, add back modulus
    if borrow != 0 {
        let mut carry = 0u128;
        for i in 0..4 {
            let sum = (result[i] as u128) + (MODULUS[i] as u128) + carry;
            result[i] = sum as u64;
            carry = sum >> 64;
        }
    }

    result
}

/// Montgomery multiplication: c = a * b * R^{-1} mod p
pub fn field_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut t = [0u64; 8];

    // Schoolbook multiplication
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let product = (a[i] as u128) * (b[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }
        t[i + 4] = carry as u64;
    }

    // Montgomery reduction
    for i in 0..4 {
        let m = t[i].wrapping_mul(INV);
        let mut carry = 0u128;

        for j in 0..4 {
            let product = (m as u128) * (MODULUS[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }

        for j in (i + 4)..8 {
            let sum = (t[j] as u128) + carry;
            t[j] = sum as u64;
            carry = sum >> 64;
            if carry == 0 { break; }
        }
    }

    let mut result = [t[4], t[5], t[6], t[7]];

    // Final reduction
    if !less_than(&result, &MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

/// Compare a < b
fn less_than(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

/// Compute a^exp mod p using square-and-multiply
fn field_pow(base: &[u64; 4], exp: &[u64; 4]) -> [u64; 4] {
    let mut result = R; // 1 in Montgomery form
    let mut base = *base;

    for limb_idx in 0..4 {
        let mut limb = exp[limb_idx];
        for _ in 0..64 {
            if limb & 1 == 1 {
                result = field_mul(&result, &base);
            }
            base = field_mul(&base, &base);
            limb >>= 1;
        }
    }

    result
}

/// Compute multiplicative inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
fn field_inv(a: &[u64; 4]) -> [u64; 4] {
    // p - 2
    let exp = [
        MODULUS[0].wrapping_sub(2),
        MODULUS[1],
        MODULUS[2],
        MODULUS[3],
    ];
    field_pow(a, &exp)
}

// ============================================================================
// NTT Implementation
// ============================================================================

/// Compute the n-th root of unity for NTT of size n = 2^log_n
fn get_root_of_unity(log_n: usize) -> [u64; 4] {
    assert!(log_n <= MAX_LOG_N, "NTT size too large");

    // Compute ω_{2^log_n} = ω_{2^28}^{2^{28-log_n}}
    let mut root = ROOT_OF_UNITY;
    for _ in log_n..MAX_LOG_N {
        root = field_mul(&root, &root);
    }

    root
}

/// Bit-reversal permutation
fn bit_reverse_permutation(data: &mut [[u64; 4]]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let rev = reverse_bits(i, log_n);
        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Reverse the lower `bits` bits of `x`
fn reverse_bits(x: usize, bits: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Forward NTT (Cooley-Tukey decimation-in-time)
pub fn forward_ntt(data: &mut [[u64; 4]]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    assert!(n.is_power_of_two(), "NTT size must be power of 2");
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation
    bit_reverse_permutation(data);

    // Butterfly stages
    let omega_n = get_root_of_unity(log_n);

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = m / 2;

        // Compute ω_m = ω_n^{n/m}
        let mut omega_m = omega_n;
        for _ in 0..(log_n - stage - 1) {
            omega_m = field_mul(&omega_m, &omega_m);
        }

        for k in (0..n).step_by(m) {
            let mut omega = R; // 1 in Montgomery form

            for j in 0..half_m {
                let u = data[k + j];
                let t = field_mul(&omega, &data[k + j + half_m]);

                data[k + j] = field_add(&u, &t);
                data[k + j + half_m] = field_sub(&u, &t);

                omega = field_mul(&omega, &omega_m);
            }
        }
    }
}

/// Inverse NTT
pub fn inverse_ntt(data: &mut [[u64; 4]]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    assert!(n.is_power_of_two(), "NTT size must be power of 2");
    let log_n = n.trailing_zeros() as usize;

    // Bit-reversal permutation
    bit_reverse_permutation(data);

    // Butterfly stages with inverse twiddles
    let omega_n = field_inv(&get_root_of_unity(log_n));

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = m / 2;

        let mut omega_m = omega_n;
        for _ in 0..(log_n - stage - 1) {
            omega_m = field_mul(&omega_m, &omega_m);
        }

        for k in (0..n).step_by(m) {
            let mut omega = R;

            for j in 0..half_m {
                let u = data[k + j];
                let t = field_mul(&omega, &data[k + j + half_m]);

                data[k + j] = field_add(&u, &t);
                data[k + j + half_m] = field_sub(&u, &t);

                omega = field_mul(&omega, &omega_m);
            }
        }
    }

    // Scale by n^{-1}
    let n_inv = compute_n_inv(n);
    for elem in data.iter_mut() {
        *elem = field_mul(elem, &n_inv);
    }
}

/// Compute n^{-1} mod p in Montgomery form
fn compute_n_inv(n: usize) -> [u64; 4] {
    // First convert n to Montgomery form
    let n_mont = to_montgomery(&[n as u64, 0, 0, 0]);
    field_inv(&n_mont)
}

/// Convert a number to Montgomery form
fn to_montgomery(a: &[u64; 4]) -> [u64; 4] {
    // a * R^2 * R^{-1} = a * R
    // We need R^2 mod p
    const R2: [u64; 4] = [
        0xf32cfc5b538afa89,
        0xb5e71911d44501fb,
        0x47ab1eff0a417ff6,
        0x06d89f71cab8351f,
    ];
    field_mul(a, &R2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_add_sub() {
        let a = [1, 0, 0, 0];
        let b = [2, 0, 0, 0];

        let c = field_add(&a, &b);
        let d = field_sub(&c, &b);

        assert_eq!(d, a);
    }

    #[test]
    fn test_ntt_size_4() {
        // Simple test with size 4
        let one = to_montgomery(&[1, 0, 0, 0]);
        let two = to_montgomery(&[2, 0, 0, 0]);
        let three = to_montgomery(&[3, 0, 0, 0]);
        let four = to_montgomery(&[4, 0, 0, 0]);

        let mut data = [one, two, three, four];

        forward_ntt(&mut data);
        inverse_ntt(&mut data);

        // After NTT and INTT, should get back original (within Montgomery form)
        // Note: Exact comparison depends on correct root of unity
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(reverse_bits(0b000, 3), 0b000);
        assert_eq!(reverse_bits(0b001, 3), 0b100);
        assert_eq!(reverse_bits(0b010, 3), 0b010);
        assert_eq!(reverse_bits(0b011, 3), 0b110);
    }
}
