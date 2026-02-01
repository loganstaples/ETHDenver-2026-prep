// HELIX GKR Prover - Metal Compute Shaders
//
// Comprehensive GPU-accelerated field and elliptic curve arithmetic for
// ZK proof generation on Apple Metal.
//
// BN254 curve parameters:
// - Scalar field (Fr): p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// - Base field (Fq): q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// - Curve equation: y² = x³ + 3 over Fq

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Type Definitions
// ============================================================================

// BN254 scalar field element represented as 4 uint64_t limbs (256 bits)
struct FieldElement {
    uint64_t limbs[4];
};

// BN254 G1 affine point
struct AffinePoint {
    FieldElement x;
    FieldElement y;
    uint32_t infinity;  // 1 if point at infinity, 0 otherwise
    uint32_t padding[3];
};

// BN254 G1 projective (Jacobian) point: (X, Y, Z) represents (X/Z², Y/Z³)
struct ProjectivePoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
};

// BN254 G2 affine point (over Fq²)
struct G2AffinePoint {
    FieldElement x_c0;  // Real part of x
    FieldElement x_c1;  // Imaginary part of x
    FieldElement y_c0;  // Real part of y
    FieldElement y_c1;  // Imaginary part of y
    uint32_t infinity;
    uint32_t padding[3];
};

// ============================================================================
// Field Constants (BN254 Scalar Field Fr)
// ============================================================================

// Scalar field modulus p
constant uint64_t FR_MODULUS[4] = {
    0x43e1f593f0000001ULL,  // limb 0 (least significant)
    0x2833e84879b97091ULL,  // limb 1
    0xb85045b68181585dULL,  // limb 2
    0x30644e72e131a029ULL   // limb 3 (most significant)
};

// R = 2^256 mod p (Montgomery constant)
constant uint64_t FR_R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

// R² mod p (for converting to Montgomery form)
constant uint64_t FR_R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL
};

// Montgomery reduction parameter: -p^(-1) mod 2^64
constant uint64_t FR_INV = 0xc2e1f593efffffffULL;

// ============================================================================
// Base Field Constants (BN254 Fq)
// ============================================================================

// Base field modulus q
constant uint64_t FQ_MODULUS[4] = {
    0x3c208c16d87cfd47ULL,
    0x97816a916871ca8dULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

// R = 2^256 mod q
constant uint64_t FQ_R[4] = {
    0xd35d438dc58f0d9dULL,
    0xa78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0xe0a77c19a07df2fULL
};

// R² mod q
constant uint64_t FQ_R2[4] = {
    0xf32cfc5b538afa89ULL,
    0xb5e71911d44501fbULL,
    0x47ab1eff0a417ff6ULL,
    0x06d89f71cab8351fULL
};

// -q^(-1) mod 2^64
constant uint64_t FQ_INV = 0x87d20782e4866389ULL;

// Curve parameter b = 3
constant uint64_t CURVE_B[4] = {
    0x3ULL, 0x0ULL, 0x0ULL, 0x0ULL
};

// ============================================================================
// 64-bit Arithmetic Helpers
// ============================================================================

// 64x64 -> 128 bit multiplication
inline void mul64(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + p2;
    uint64_t mid_carry = (mid < p1) ? (1ULL << 32) : 0ULL;

    lo = p0 + (mid << 32);
    uint64_t lo_carry = (lo < p0) ? 1ULL : 0ULL;

    hi = p3 + (mid >> 32) + mid_carry + lo_carry;
}

// Add two 256-bit numbers with carry
inline uint64_t add256(thread FieldElement& result,
                       const thread FieldElement& a,
                       const thread FieldElement& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        result.limbs[i] = sum;
    }
    return carry;
}

// Add device element to thread element
inline uint64_t add256_device(thread FieldElement& result,
                              const thread FieldElement& a,
                              const device FieldElement& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        result.limbs[i] = sum;
    }
    return carry;
}

// Subtract two 256-bit numbers with borrow
inline uint64_t sub256(thread FieldElement& result,
                       const thread FieldElement& a,
                       const thread FieldElement& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = (a.limbs[i] < b.limbs[i]) || (borrow && a.limbs[i] == b.limbs[i]) ? 1 : 0;
        result.limbs[i] = diff;
    }
    return borrow;
}

// Compare a to constant array
inline int compare_const(const thread FieldElement& a, constant uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] < b[i]) return -1;
        if (a.limbs[i] > b[i]) return 1;
    }
    return 0;
}

// Check if a >= modulus
inline bool needs_reduction(const thread FieldElement& a, constant uint64_t mod[4]) {
    return compare_const(a, mod) >= 0;
}

// Reduce mod p (scalar field)
inline void reduce_fr(thread FieldElement& a) {
    if (needs_reduction(a, FR_MODULUS)) {
        FieldElement mod;
        for (int i = 0; i < 4; i++) mod.limbs[i] = FR_MODULUS[i];
        sub256(a, a, mod);
    }
}

// Reduce mod q (base field)
inline void reduce_fq(thread FieldElement& a) {
    if (needs_reduction(a, FQ_MODULUS)) {
        FieldElement mod;
        for (int i = 0; i < 4; i++) mod.limbs[i] = FQ_MODULUS[i];
        sub256(a, a, mod);
    }
}

// Check if element is zero
inline bool is_zero(const thread FieldElement& a) {
    return a.limbs[0] == 0 && a.limbs[1] == 0 && a.limbs[2] == 0 && a.limbs[3] == 0;
}

// Set element to zero
inline void set_zero(thread FieldElement& a) {
    a.limbs[0] = 0;
    a.limbs[1] = 0;
    a.limbs[2] = 0;
    a.limbs[3] = 0;
}

// Copy constant to thread local
inline void copy_const(thread FieldElement& dst, constant uint64_t src[4]) {
    for (int i = 0; i < 4; i++) dst.limbs[i] = src[i];
}

// ============================================================================
// Scalar Field (Fr) Operations
// ============================================================================

// Field addition: c = a + b mod p
kernel void field_add(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, local_b, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
        local_b.limbs[i] = b[id].limbs[i];
    }

    uint64_t carry = add256(result, local_a, local_b);

    FieldElement modulus;
    copy_const(modulus, FR_MODULUS);

    if (carry || needs_reduction(result, FR_MODULUS)) {
        sub256(result, result, modulus);
    }

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// Field subtraction: c = a - b mod p
kernel void field_sub(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, local_b, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
        local_b.limbs[i] = b[id].limbs[i];
    }

    uint64_t borrow = sub256(result, local_a, local_b);

    if (borrow) {
        FieldElement modulus;
        copy_const(modulus, FR_MODULUS);
        add256(result, result, modulus);
    }

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// Field negation: c = -a mod p
kernel void field_neg(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
    }

    if (is_zero(local_a)) {
        for (int i = 0; i < 4; i++) c[id].limbs[i] = 0;
        return;
    }

    FieldElement modulus, result;
    copy_const(modulus, FR_MODULUS);
    sub256(result, modulus, local_a);

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// Montgomery multiplication helper (inline)
inline void mont_mul_fr(thread FieldElement& c,
                        const thread FieldElement& a,
                        const thread FieldElement& b) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(a.limbs[i], b.limbs[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 4] = carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * FR_INV;
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(m, FR_MODULUS[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }

        for (int j = i + 4; j < 8 && carry; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1 : 0;
            t[j] = sum;
        }
    }

    // Copy result
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = t[i + 4];
    }
    reduce_fr(c);
}

// Field multiplication kernel
kernel void field_mul(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, local_b, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
        local_b.limbs[i] = b[id].limbs[i];
    }

    mont_mul_fr(result, local_a, local_b);

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// Field squaring kernel
kernel void field_square(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
    }

    // Use multiplication for now (optimized squaring would be faster)
    mont_mul_fr(result, local_a, local_a);

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// ============================================================================
// Batch Field Operations
// ============================================================================

// Batch field addition with constant
kernel void field_add_const(
    const device FieldElement* a [[buffer(0)]],
    constant FieldElement& b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, local_b, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
        local_b.limbs[i] = b.limbs[i];
    }

    uint64_t carry = add256(result, local_a, local_b);

    if (carry || needs_reduction(result, FR_MODULUS)) {
        FieldElement modulus;
        copy_const(modulus, FR_MODULUS);
        sub256(result, result, modulus);
    }

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// Batch field multiplication with constant
kernel void field_mul_const(
    const device FieldElement* a [[buffer(0)]],
    constant FieldElement& b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a, local_b, result;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
        local_b.limbs[i] = b.limbs[i];
    }

    mont_mul_fr(result, local_a, local_b);

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = result.limbs[i];
    }
}

// ============================================================================
// Batch Inversion (Montgomery's Trick Prep)
// ============================================================================

// Compute partial products for batch inversion
// products[i] = a[0] * a[1] * ... * a[i]
kernel void batch_inv_products(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* products [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    constant uint32_t& chunk_size [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint32_t start = id * chunk_size;
    uint32_t end = min(start + chunk_size, n);

    if (start >= n) return;

    FieldElement acc;
    if (start == 0) {
        for (int i = 0; i < 4; i++) acc.limbs[i] = a[0].limbs[i];
    } else {
        // Would need to combine with previous chunk
        for (int i = 0; i < 4; i++) acc.limbs[i] = a[start].limbs[i];
    }

    for (uint32_t i = start; i < end; i++) {
        if (i > start) {
            FieldElement local_a;
            for (int j = 0; j < 4; j++) local_a.limbs[j] = a[i].limbs[j];
            mont_mul_fr(acc, acc, local_a);
        }
        for (int j = 0; j < 4; j++) products[i].limbs[j] = acc.limbs[j];
    }
}

// Compute final inverses from products and inverse of total product
kernel void batch_inv_finalize(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* products [[buffer(1)]],
    constant FieldElement& total_inv [[buffer(2)]],
    device FieldElement* inverses [[buffer(3)]],
    constant uint32_t& n [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    FieldElement result;
    FieldElement local_inv;
    for (int i = 0; i < 4; i++) local_inv.limbs[i] = total_inv.limbs[i];

    if (id == 0) {
        // inverses[0] = total_inv * products[n-1] / a[0] = ... = 1/a[0]
        // This needs special handling in the full algorithm
        for (int i = 0; i < 4; i++) result.limbs[i] = local_inv.limbs[i];
    } else {
        // inverses[i] = products[i-1] * inv_acc
        FieldElement prev_prod;
        for (int i = 0; i < 4; i++) prev_prod.limbs[i] = products[id - 1].limbs[i];
        mont_mul_fr(result, prev_prod, local_inv);
    }

    for (int i = 0; i < 4; i++) inverses[id].limbs[i] = result.limbs[i];
}

// ============================================================================
// Elliptic Curve Operations (BN254 G1)
// ============================================================================

// Check if projective point is at infinity
inline bool point_is_identity(const thread ProjectivePoint& p) {
    return is_zero(p.z);
}

// Point doubling in Jacobian coordinates
// Cost: 3M + 4S + 8add (for a = 0 curve like BN254)
inline void point_double(thread ProjectivePoint& r, const thread ProjectivePoint& p) {
    if (point_is_identity(p)) {
        r = p;
        return;
    }

    FieldElement a, b, c, d, e, f;

    // A = Y1²
    mont_mul_fr(a, p.y, p.y);

    // B = 4 * X1 * A
    mont_mul_fr(b, p.x, a);
    FieldElement two_b;
    uint64_t carry = add256(two_b, b, b);
    if (carry || needs_reduction(two_b, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(two_b, two_b, mod);
    }
    carry = add256(b, two_b, two_b);
    if (carry || needs_reduction(b, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(b, b, mod);
    }

    // C = 8 * A²
    mont_mul_fr(c, a, a);  // A²
    carry = add256(c, c, c);  // 2A²
    if (carry || needs_reduction(c, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(c, c, mod);
    }
    carry = add256(c, c, c);  // 4A²
    if (carry || needs_reduction(c, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(c, c, mod);
    }
    carry = add256(c, c, c);  // 8A²
    if (carry || needs_reduction(c, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(c, c, mod);
    }

    // D = 3 * X1² (since a = 0 for BN254)
    mont_mul_fr(d, p.x, p.x);  // X1²
    FieldElement three_d;
    carry = add256(three_d, d, d);  // 2X1²
    if (carry || needs_reduction(three_d, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(three_d, three_d, mod);
    }
    carry = add256(d, three_d, d);  // 3X1²
    if (carry || needs_reduction(d, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(d, d, mod);
    }

    // X3 = D² - 2B
    mont_mul_fr(r.x, d, d);  // D²
    sub256(r.x, r.x, b);     // D² - B
    if (needs_reduction(r.x, FR_MODULUS)) {
        // Went negative, add modulus
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.x, r.x, mod);
    }
    sub256(r.x, r.x, b);     // D² - 2B
    if (needs_reduction(r.x, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.x, r.x, mod);
    }

    // Y3 = D * (B - X3) - C
    sub256(f, b, r.x);
    if (needs_reduction(f, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(f, f, mod);
    }
    mont_mul_fr(r.y, d, f);
    sub256(r.y, r.y, c);
    if (needs_reduction(r.y, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.y, r.y, mod);
    }

    // Z3 = 2 * Y1 * Z1
    mont_mul_fr(r.z, p.y, p.z);
    carry = add256(r.z, r.z, r.z);
    if (carry || needs_reduction(r.z, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(r.z, r.z, mod);
    }
}

// Mixed addition: P (projective) + Q (affine) -> R (projective)
// Cost: 7M + 4S + 9add
inline void point_add_mixed(thread ProjectivePoint& r,
                            const thread ProjectivePoint& p,
                            const thread AffinePoint& q) {
    if (q.infinity) {
        r = p;
        return;
    }

    if (point_is_identity(p)) {
        // Convert affine to projective
        r.x = q.x;
        r.y = q.y;
        copy_const(r.z, FR_R);  // Z = 1 in Montgomery form
        return;
    }

    FieldElement z1_sq, u2, z1_cu, s2, h, hh, i, j, rr, v;

    // Z1²
    mont_mul_fr(z1_sq, p.z, p.z);

    // U2 = X2 * Z1²
    mont_mul_fr(u2, q.x, z1_sq);

    // Z1³
    mont_mul_fr(z1_cu, z1_sq, p.z);

    // S2 = Y2 * Z1³
    mont_mul_fr(s2, q.y, z1_cu);

    // H = U2 - X1
    sub256(h, u2, p.x);
    if (needs_reduction(h, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(h, h, mod);
    }

    // HH = H²
    mont_mul_fr(hh, h, h);

    // I = 4 * HH
    uint64_t carry = add256(i, hh, hh);
    if (carry || needs_reduction(i, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(i, i, mod);
    }
    carry = add256(i, i, i);
    if (carry || needs_reduction(i, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(i, i, mod);
    }

    // J = H * I
    mont_mul_fr(j, h, i);

    // rr = 2 * (S2 - Y1)
    sub256(rr, s2, p.y);
    if (needs_reduction(rr, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(rr, rr, mod);
    }
    carry = add256(rr, rr, rr);
    if (carry || needs_reduction(rr, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(rr, rr, mod);
    }

    // V = X1 * I
    mont_mul_fr(v, p.x, i);

    // X3 = rr² - J - 2V
    FieldElement rr_sq;
    mont_mul_fr(rr_sq, rr, rr);
    sub256(r.x, rr_sq, j);
    if (needs_reduction(r.x, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.x, r.x, mod);
    }
    sub256(r.x, r.x, v);
    if (needs_reduction(r.x, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.x, r.x, mod);
    }
    sub256(r.x, r.x, v);
    if (needs_reduction(r.x, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.x, r.x, mod);
    }

    // Y3 = rr * (V - X3) - 2 * Y1 * J
    FieldElement temp;
    sub256(temp, v, r.x);
    if (needs_reduction(temp, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(temp, temp, mod);
    }
    mont_mul_fr(r.y, rr, temp);
    mont_mul_fr(temp, p.y, j);
    carry = add256(temp, temp, temp);
    if (carry || needs_reduction(temp, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(temp, temp, mod);
    }
    sub256(r.y, r.y, temp);
    if (needs_reduction(r.y, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.y, r.y, mod);
    }

    // Z3 = (Z1 + H)² - Z1² - HH
    carry = add256(temp, p.z, h);
    if (carry || needs_reduction(temp, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        sub256(temp, temp, mod);
    }
    mont_mul_fr(r.z, temp, temp);
    sub256(r.z, r.z, z1_sq);
    if (needs_reduction(r.z, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.z, r.z, mod);
    }
    sub256(r.z, r.z, hh);
    if (needs_reduction(r.z, FR_MODULUS)) {
        FieldElement mod; copy_const(mod, FR_MODULUS);
        add256(r.z, r.z, mod);
    }
}

// Point doubling kernel
kernel void ec_point_double(
    const device ProjectivePoint* p [[buffer(0)]],
    device ProjectivePoint* r [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    ProjectivePoint local_p, result;
    for (int i = 0; i < 4; i++) {
        local_p.x.limbs[i] = p[id].x.limbs[i];
        local_p.y.limbs[i] = p[id].y.limbs[i];
        local_p.z.limbs[i] = p[id].z.limbs[i];
    }

    point_double(result, local_p);

    for (int i = 0; i < 4; i++) {
        r[id].x.limbs[i] = result.x.limbs[i];
        r[id].y.limbs[i] = result.y.limbs[i];
        r[id].z.limbs[i] = result.z.limbs[i];
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

// Parallel sum reduction (tree-based)
kernel void parallel_sum(
    device FieldElement* elements [[buffer(0)]],
    constant uint32_t& n [[buffer(1)]],
    constant uint32_t& stride [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint32_t idx = id * stride * 2;
    uint32_t pair_idx = idx + stride;

    if (pair_idx >= n) return;

    FieldElement a, b, result;
    for (int i = 0; i < 4; i++) {
        a.limbs[i] = elements[idx].limbs[i];
        b.limbs[i] = elements[pair_idx].limbs[i];
    }

    uint64_t carry = add256(result, a, b);

    if (carry || needs_reduction(result, FR_MODULUS)) {
        FieldElement mod;
        copy_const(mod, FR_MODULUS);
        sub256(result, result, mod);
    }

    for (int i = 0; i < 4; i++) {
        elements[idx].limbs[i] = result.limbs[i];
    }
}

// Parallel product (for commitments)
kernel void parallel_product(
    device FieldElement* elements [[buffer(0)]],
    constant uint32_t& n [[buffer(1)]],
    constant uint32_t& stride [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint32_t idx = id * stride * 2;
    uint32_t pair_idx = idx + stride;

    if (pair_idx >= n) return;

    FieldElement a, b, result;
    for (int i = 0; i < 4; i++) {
        a.limbs[i] = elements[idx].limbs[i];
        b.limbs[i] = elements[pair_idx].limbs[i];
    }

    mont_mul_fr(result, a, b);

    for (int i = 0; i < 4; i++) {
        elements[idx].limbs[i] = result.limbs[i];
    }
}

// ============================================================================
// Polynomial Evaluation
// ============================================================================

// Polynomial evaluation step for multilinear extension
// new[i] = old[i] + r * (old[i + half_size] - old[i])
kernel void poly_eval_step(
    const device FieldElement* old_evals [[buffer(0)]],
    constant FieldElement& r [[buffer(1)]],
    device FieldElement* new_evals [[buffer(2)]],
    constant uint32_t& half_size [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= half_size) return;

    FieldElement f0, f1, local_r, diff, scaled, result;

    for (int i = 0; i < 4; i++) {
        f0.limbs[i] = old_evals[id].limbs[i];
        f1.limbs[i] = old_evals[id + half_size].limbs[i];
        local_r.limbs[i] = r.limbs[i];
    }

    // diff = f1 - f0
    sub256(diff, f1, f0);
    if (needs_reduction(diff, FR_MODULUS)) {
        FieldElement mod;
        copy_const(mod, FR_MODULUS);
        add256(diff, diff, mod);
    }

    // scaled = r * diff
    mont_mul_fr(scaled, local_r, diff);

    // result = f0 + scaled
    uint64_t carry = add256(result, f0, scaled);
    if (carry || needs_reduction(result, FR_MODULUS)) {
        FieldElement mod;
        copy_const(mod, FR_MODULUS);
        sub256(result, result, mod);
    }

    for (int i = 0; i < 4; i++) {
        new_evals[id].limbs[i] = result.limbs[i];
    }
}

// ============================================================================
// Matrix Operations (for Neural Network Layers)
// ============================================================================

// Matrix-vector multiplication: c = A * x
// Each thread computes one element of the result
kernel void matmul_field(
    const device FieldElement* A [[buffer(0)]],  // rows x cols matrix
    const device FieldElement* x [[buffer(1)]],  // cols vector
    device FieldElement* c [[buffer(2)]],        // rows result
    constant uint32_t& rows [[buffer(3)]],
    constant uint32_t& cols [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= rows) return;

    FieldElement sum;
    set_zero(sum);

    for (uint32_t j = 0; j < cols; j++) {
        FieldElement a_ij, x_j, prod;
        for (int k = 0; k < 4; k++) {
            a_ij.limbs[k] = A[id * cols + j].limbs[k];
            x_j.limbs[k] = x[j].limbs[k];
        }

        mont_mul_fr(prod, a_ij, x_j);

        uint64_t carry = add256(sum, sum, prod);
        if (carry || needs_reduction(sum, FR_MODULUS)) {
            FieldElement mod;
            copy_const(mod, FR_MODULUS);
            sub256(sum, sum, mod);
        }
    }

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = sum.limbs[i];
    }
}

// Tiled matrix-vector multiplication for better cache utilization
kernel void matmul_field_tiled(
    const device FieldElement* A [[buffer(0)]],
    const device FieldElement* x [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    constant uint32_t& rows [[buffer(3)]],
    constant uint32_t& cols [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    threadgroup FieldElement* shared_x [[threadgroup(0)]]
) {
    if (id >= rows) return;

    const uint32_t TILE_SIZE = 64;

    FieldElement sum;
    set_zero(sum);

    // Process tiles
    for (uint32_t tile_start = 0; tile_start < cols; tile_start += TILE_SIZE) {
        // Load tile of x into shared memory
        if (local_id < TILE_SIZE && tile_start + local_id < cols) {
            for (int k = 0; k < 4; k++) {
                shared_x[local_id].limbs[k] = x[tile_start + local_id].limbs[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        uint32_t tile_end = min(tile_start + TILE_SIZE, cols);
        for (uint32_t j = tile_start; j < tile_end; j++) {
            FieldElement a_ij, x_j, prod;
            for (int k = 0; k < 4; k++) {
                a_ij.limbs[k] = A[id * cols + j].limbs[k];
                x_j.limbs[k] = shared_x[j - tile_start].limbs[k];
            }

            mont_mul_fr(prod, a_ij, x_j);

            uint64_t carry = add256(sum, sum, prod);
            if (carry || needs_reduction(sum, FR_MODULUS)) {
                FieldElement mod;
                copy_const(mod, FR_MODULUS);
                sub256(sum, sum, mod);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = 0; i < 4; i++) {
        c[id].limbs[i] = sum.limbs[i];
    }
}

// Element-wise ReLU approximation (returns 0 if negative bit is set)
// This is an approximation - true ReLU needs comparison to field element
kernel void relu_approx(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a;
    for (int i = 0; i < 4; i++) {
        local_a.limbs[i] = a[id].limbs[i];
    }

    // Simple approximation: if the high bit of the high limb is set,
    // the value is in the upper half of the field (representing negative)
    bool is_negative = (local_a.limbs[3] >> 63) != 0;

    if (is_negative) {
        for (int i = 0; i < 4; i++) c[id].limbs[i] = 0;
    } else {
        for (int i = 0; i < 4; i++) c[id].limbs[i] = local_a.limbs[i];
    }
}
