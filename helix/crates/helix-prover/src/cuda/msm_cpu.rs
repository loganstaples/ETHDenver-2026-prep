//! CPU fallback for MSM using Pippenger's algorithm.
//!
//! This implements multi-scalar multiplication over BN254 G1 curve
//! using the Pippenger bucket method with proper field arithmetic.

use super::ntt_cpu::{field_add, field_sub, field_mul, MODULUS};

/// BN254 base field modulus (for Fq, not Fr)
const FQ_MODULUS: [u64; 4] = [
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// Montgomery R for Fq
const FQ_R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

/// -Fq^{-1} mod 2^64
const FQ_INV: u64 = 0x87d20782e4866389;

/// Affine point on BN254 G1.
#[derive(Clone, Copy, Debug)]
struct AffinePoint {
    x: [u64; 4],
    y: [u64; 4],
    infinity: bool,
}

/// Projective point on BN254 G1 (Jacobian coordinates).
#[derive(Clone, Copy, Debug)]
struct ProjectivePoint {
    x: [u64; 4],
    y: [u64; 4],
    z: [u64; 4],
}

impl ProjectivePoint {
    fn identity() -> Self {
        Self {
            x: FQ_R,
            y: FQ_R,
            z: [0, 0, 0, 0],
        }
    }

    fn is_identity(&self) -> bool {
        self.z[0] == 0 && self.z[1] == 0 && self.z[2] == 0 && self.z[3] == 0
    }

    fn from_affine(p: &AffinePoint) -> Self {
        if p.infinity {
            Self::identity()
        } else {
            Self {
                x: p.x,
                y: p.y,
                z: FQ_R, // Z = 1 in Montgomery form
            }
        }
    }
}

// ============================================================================
// Fq Field Arithmetic (base field for curve points)
// ============================================================================

fn fq_add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u128;

    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + carry;
        result[i] = sum as u64;
        carry = sum >> 64;
    }

    // Reduce if >= modulus
    if carry != 0 || !less_than(&result, &FQ_MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (FQ_MODULUS[i] as i128) - borrow;
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

fn fq_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
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
            let sum = (result[i] as u128) + (FQ_MODULUS[i] as u128) + carry;
            result[i] = sum as u64;
            carry = sum >> 64;
        }
    }

    result
}

fn fq_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    // Montgomery multiplication
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
        let m = t[i].wrapping_mul(FQ_INV);
        let mut carry = 0u128;

        for j in 0..4 {
            let product = (m as u128) * (FQ_MODULUS[j] as u128) + (t[i + j] as u128) + carry;
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
    if !less_than(&result, &FQ_MODULUS) {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (FQ_MODULUS[i] as i128) - borrow;
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

fn fq_double(a: &[u64; 4]) -> [u64; 4] {
    fq_add(a, a)
}

fn less_than(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false
}

// ============================================================================
// Point Arithmetic (Jacobian Coordinates)
// ============================================================================

fn point_double(p: &ProjectivePoint) -> ProjectivePoint {
    if p.is_identity() {
        return *p;
    }

    // A = Y^2
    let a = fq_mul(&p.y, &p.y);

    // B = 4*X*A
    let xa = fq_mul(&p.x, &a);
    let b = fq_double(&fq_double(&xa));

    // C = 8*A^2
    let aa = fq_mul(&a, &a);
    let c = fq_double(&fq_double(&fq_double(&aa)));

    // D = 3*X^2 (a=0 for BN254, so curve is y^2 = x^3 + b)
    let xx = fq_mul(&p.x, &p.x);
    let d = fq_add(&fq_double(&xx), &xx);

    // X3 = D^2 - 2*B
    let dd = fq_mul(&d, &d);
    let x3 = fq_sub(&dd, &fq_double(&b));

    // Y3 = D*(B - X3) - C
    let b_minus_x3 = fq_sub(&b, &x3);
    let d_times_diff = fq_mul(&d, &b_minus_x3);
    let y3 = fq_sub(&d_times_diff, &c);

    // Z3 = 2*Y*Z
    let yz = fq_mul(&p.y, &p.z);
    let z3 = fq_double(&yz);

    ProjectivePoint { x: x3, y: y3, z: z3 }
}

fn point_add_mixed(p: &ProjectivePoint, q: &AffinePoint) -> ProjectivePoint {
    if q.infinity {
        return *p;
    }

    if p.is_identity() {
        return ProjectivePoint::from_affine(q);
    }

    // Z1^2
    let z1_sq = fq_mul(&p.z, &p.z);

    // U2 = X2 * Z1^2
    let u2 = fq_mul(&q.x, &z1_sq);

    // Z1^3
    let z1_cu = fq_mul(&z1_sq, &p.z);

    // S2 = Y2 * Z1^3
    let s2 = fq_mul(&q.y, &z1_cu);

    // H = U2 - X1
    let h = fq_sub(&u2, &p.x);

    // R = S2 - Y1
    let r = fq_sub(&s2, &p.y);

    // Check if H == 0 (points have same x-coordinate)
    if h == [0; 4] {
        if r == [0; 4] {
            // Points are the same, do doubling
            return point_double(p);
        } else {
            // Points are negatives, return identity
            return ProjectivePoint::identity();
        }
    }

    // HH = H^2
    let hh = fq_mul(&h, &h);

    // HHH = H^3
    let hhh = fq_mul(&hh, &h);

    // V = X1 * HH
    let v = fq_mul(&p.x, &hh);

    // X3 = R^2 - HHH - 2*V
    let rr = fq_mul(&r, &r);
    let x3 = fq_sub(&fq_sub(&rr, &hhh), &fq_double(&v));

    // Y3 = R*(V - X3) - Y1*HHH
    let v_minus_x3 = fq_sub(&v, &x3);
    let r_times_diff = fq_mul(&r, &v_minus_x3);
    let y1_hhh = fq_mul(&p.y, &hhh);
    let y3 = fq_sub(&r_times_diff, &y1_hhh);

    // Z3 = Z1 * H
    let z3 = fq_mul(&p.z, &h);

    ProjectivePoint { x: x3, y: y3, z: z3 }
}

fn point_add_proj(p: &ProjectivePoint, q: &ProjectivePoint) -> ProjectivePoint {
    if p.is_identity() {
        return *q;
    }
    if q.is_identity() {
        return *p;
    }

    let z1_sq = fq_mul(&p.z, &p.z);
    let z2_sq = fq_mul(&q.z, &q.z);
    let z1_cu = fq_mul(&z1_sq, &p.z);
    let z2_cu = fq_mul(&z2_sq, &q.z);

    let u1 = fq_mul(&p.x, &z2_sq);
    let u2 = fq_mul(&q.x, &z1_sq);
    let s1 = fq_mul(&p.y, &z2_cu);
    let s2 = fq_mul(&q.y, &z1_cu);

    let h = fq_sub(&u2, &u1);
    let r = fq_sub(&s2, &s1);

    if h == [0; 4] {
        if r == [0; 4] {
            return point_double(p);
        } else {
            return ProjectivePoint::identity();
        }
    }

    let hh = fq_mul(&h, &h);
    let hhh = fq_mul(&hh, &h);
    let v = fq_mul(&u1, &hh);

    let rr = fq_mul(&r, &r);
    let x3 = fq_sub(&fq_sub(&rr, &hhh), &fq_double(&v));

    let v_minus_x3 = fq_sub(&v, &x3);
    let r_times_diff = fq_mul(&r, &v_minus_x3);
    let s1_hhh = fq_mul(&s1, &hhh);
    let y3 = fq_sub(&r_times_diff, &s1_hhh);

    let z1z2 = fq_mul(&p.z, &q.z);
    let z3 = fq_mul(&z1z2, &h);

    ProjectivePoint { x: x3, y: y3, z: z3 }
}

// ============================================================================
// Scalar Operations (on Fr, the scalar field)
// ============================================================================

fn scalar_bit(scalar: &[u64; 4], bit: usize) -> bool {
    let limb = bit / 64;
    let bit_in_limb = bit % 64;
    if limb >= 4 { return false; }
    (scalar[limb] >> bit_in_limb) & 1 == 1
}

fn get_window(scalar: &[u64; 4], window_idx: usize, window_size: usize) -> usize {
    let bit_offset = window_idx * window_size;
    let limb_idx = bit_offset / 64;
    let bit_in_limb = bit_offset % 64;

    if limb_idx >= 4 { return 0; }

    let mut value = scalar[limb_idx] >> bit_in_limb;

    let bits_from_first = 64 - bit_in_limb;
    if bits_from_first < window_size && limb_idx + 1 < 4 {
        let remaining_bits = window_size - bits_from_first;
        let mask = (1u64 << remaining_bits) - 1;
        value |= (scalar[limb_idx + 1] & mask) << bits_from_first;
    }

    let mask = (1usize << window_size) - 1;
    (value as usize) & mask
}

// ============================================================================
// Pippenger's Algorithm
// ============================================================================

/// Compute MSM using Pippenger's bucket method.
pub fn compute_msm(
    points: &[[u64; 8]],
    scalars: &[[u64; 4]],
    window_size: usize,
) -> [u64; 12] {
    if points.is_empty() {
        return [0u64; 12];
    }

    let window_size = window_size.clamp(4, 16);
    let num_windows = (256 + window_size - 1) / window_size;
    let num_buckets = (1 << window_size) - 1;

    // Convert points to AffinePoint structs
    let affine_points: Vec<AffinePoint> = points.iter().map(|p| {
        AffinePoint {
            x: [p[0], p[1], p[2], p[3]],
            y: [p[4], p[5], p[6], p[7]],
            infinity: p[0] == 0 && p[1] == 0 && p[2] == 0 && p[3] == 0
                   && p[4] == 0 && p[5] == 0 && p[6] == 0 && p[7] == 0,
        }
    }).collect();

    // Process windows from most significant to least significant
    let mut result = ProjectivePoint::identity();

    for w in (0..num_windows).rev() {
        // Double result `window_size` times
        for _ in 0..window_size {
            result = point_double(&result);
        }

        // Accumulate points into buckets
        let mut buckets: Vec<ProjectivePoint> = vec![ProjectivePoint::identity(); num_buckets];

        for (i, (point, scalar)) in affine_points.iter().zip(scalars.iter()).enumerate() {
            let bucket_idx = get_window(scalar, w, window_size);
            if bucket_idx > 0 {
                buckets[bucket_idx - 1] = point_add_mixed(&buckets[bucket_idx - 1], point);
            }
        }

        // Reduce buckets: running sum technique
        let mut running_sum = ProjectivePoint::identity();
        let mut window_sum = ProjectivePoint::identity();

        for bucket_idx in (0..num_buckets).rev() {
            running_sum = point_add_proj(&running_sum, &buckets[bucket_idx]);
            window_sum = point_add_proj(&window_sum, &running_sum);
        }

        result = point_add_proj(&result, &window_sum);
    }

    // Convert to limb array
    [
        result.x[0], result.x[1], result.x[2], result.x[3],
        result.y[0], result.y[1], result.y[2], result.y[3],
        result.z[0], result.z[1], result.z[2], result.z[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fq_arithmetic() {
        // Test field addition
        let a = [1, 0, 0, 0];
        let b = [2, 0, 0, 0];
        let c = fq_add(&a, &b);
        assert_eq!(c, [3, 0, 0, 0]);

        // Test field subtraction
        let d = fq_sub(&c, &b);
        assert_eq!(d, a);
    }

    #[test]
    fn test_point_identity() {
        let id = ProjectivePoint::identity();
        assert!(id.is_identity());

        let doubled = point_double(&id);
        assert!(doubled.is_identity());
    }

    #[test]
    fn test_msm_empty() {
        let result = compute_msm(&[], &[], 8);
        assert_eq!(result, [0u64; 12]);
    }
}
