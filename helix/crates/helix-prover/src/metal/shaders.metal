// HELIX GKR Prover - Metal Compute Shaders
//
// These shaders implement GPU-accelerated field arithmetic for the BN254
// scalar field used in GKR proofs.
//
// BN254 scalar field modulus:
// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617

#include <metal_stdlib>
using namespace metal;

// BN254 scalar field element represented as 4 uint64_t limbs (256 bits)
struct FieldElement {
    uint64_t limbs[4];
};

// Field modulus p
constant uint64_t MODULUS[4] = {
    0x43e1f593f0000001ULL,  // limb 0 (least significant)
    0x2833e84879b97091ULL,  // limb 1
    0xb85045b68181585dULL,  // limb 2
    0x30644e72e131a029ULL   // limb 3 (most significant)
};

// R = 2^256 mod p (Montgomery constant)
constant uint64_t R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

// R^2 mod p
constant uint64_t R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL
};

// Montgomery parameter: -p^(-1) mod 2^64
constant uint64_t INV = 0xc2e1f593effffffULL;

// Helper: Add two 256-bit numbers, return carry
inline uint64_t add_with_carry(thread FieldElement& result,
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

// Helper: Subtract two 256-bit numbers, return borrow
inline uint64_t sub_with_borrow(thread FieldElement& result,
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

// Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
inline int compare(const thread FieldElement& a, constant uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] < b[i]) return -1;
        if (a.limbs[i] > b[i]) return 1;
    }
    return 0;
}

// Reduce mod p
inline void reduce(thread FieldElement& a) {
    if (compare(a, MODULUS) >= 0) {
        FieldElement modulus;
        for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];
        sub_with_borrow(a, a, modulus);
    }
}

// Field addition: c = a + b mod p
kernel void field_add(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a = a[id];
    FieldElement local_b = b[id];
    FieldElement result;

    uint64_t carry = add_with_carry(result, local_a, local_b);

    // If carry or result >= p, subtract p
    FieldElement modulus;
    for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];

    if (carry || compare(result, MODULUS) >= 0) {
        sub_with_borrow(result, result, modulus);
    }

    c[id] = result;
}

// Field subtraction: c = a - b mod p
kernel void field_sub(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a = a[id];
    FieldElement local_b = b[id];
    FieldElement result;

    uint64_t borrow = sub_with_borrow(result, local_a, local_b);

    // If borrow, add p
    if (borrow) {
        FieldElement modulus;
        for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];
        add_with_carry(result, result, modulus);
    }

    c[id] = result;
}

// Field negation: c = -a mod p
kernel void field_neg(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a = a[id];

    // Check if a is zero
    bool is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (local_a.limbs[i] != 0) {
            is_zero = false;
            break;
        }
    }

    if (is_zero) {
        c[id] = local_a;
        return;
    }

    // -a = p - a
    FieldElement modulus;
    for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];

    FieldElement result;
    sub_with_borrow(result, modulus, local_a);
    c[id] = result;
}

// 64x64 -> 128 bit multiplication
inline void mul64(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + p2;
    uint64_t mid_carry = (mid < p1) ? 1ULL << 32 : 0;

    lo = p0 + (mid << 32);
    uint64_t lo_carry = (lo < p0) ? 1 : 0;

    hi = p3 + (mid >> 32) + mid_carry + lo_carry;
}

// Montgomery multiplication: c = a * b * R^(-1) mod p
kernel void field_mul(
    const device FieldElement* a [[buffer(0)]],
    const device FieldElement* b [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a = a[id];
    FieldElement local_b = b[id];

    // Result accumulator (8 limbs for 512-bit intermediate)
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(local_a.limbs[i], local_b.limbs[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 4] = carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * INV;
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(m, MODULUS[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }

        // Propagate carry
        for (int j = i + 4; j < 8 && carry; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1 : 0;
            t[j] = sum;
        }
    }

    // Result is in t[4..7]
    FieldElement result;
    for (int i = 0; i < 4; i++) {
        result.limbs[i] = t[i + 4];
    }

    // Final reduction if needed
    reduce(result);

    c[id] = result;
}

// Field squaring (optimized)
kernel void field_square(
    const device FieldElement* a [[buffer(0)]],
    device FieldElement* c [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    FieldElement local_a = a[id];

    // Use multiplication kernel for now
    // TODO: Implement optimized squaring
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(local_a.limbs[i], local_a.limbs[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 4] += carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * INV;
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(m, MODULUS[j], hi, lo);

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

    FieldElement result;
    for (int i = 0; i < 4; i++) {
        result.limbs[i] = t[i + 4];
    }

    reduce(result);
    c[id] = result;
}

// Parallel sum reduction (first stage)
// Each thread computes sum of elements[id] + elements[id + stride]
kernel void parallel_sum(
    device FieldElement* elements [[buffer(0)]],
    constant uint& stride [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint idx = id * stride * 2;
    uint pair_idx = idx + stride;

    FieldElement a = elements[idx];
    FieldElement b = elements[pair_idx];
    FieldElement result;

    uint64_t carry = add_with_carry(result, a, b);

    FieldElement modulus;
    for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];

    if (carry || compare(result, MODULUS) >= 0) {
        sub_with_borrow(result, result, modulus);
    }

    elements[idx] = result;
}

// Polynomial evaluation helper: compute new[i] = old[i] + r * (old[i+half] - old[i])
kernel void poly_eval_step(
    const device FieldElement* old_evals [[buffer(0)]],
    const device FieldElement* r [[buffer(1)]],  // Scalar, replicated for all threads
    device FieldElement* new_evals [[buffer(2)]],
    constant uint& half [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= half) return;

    FieldElement f0 = old_evals[id];
    FieldElement f1 = old_evals[id + half];
    FieldElement r_val = r[0];

    // diff = f1 - f0
    FieldElement diff;
    uint64_t borrow = sub_with_borrow(diff, f1, f0);
    if (borrow) {
        FieldElement modulus;
        for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];
        add_with_carry(diff, diff, modulus);
    }

    // scaled = r * diff (using simplified mul for now)
    // TODO: Full Montgomery multiplication
    FieldElement scaled = diff;  // Placeholder

    // result = f0 + scaled
    FieldElement result;
    uint64_t carry = add_with_carry(result, f0, scaled);

    FieldElement modulus;
    for (int i = 0; i < 4; i++) modulus.limbs[i] = MODULUS[i];

    if (carry || compare(result, MODULUS) >= 0) {
        sub_with_borrow(result, result, modulus);
    }

    new_evals[id] = result;
}

// Matrix-vector multiplication for neural network layers
// c[i] = sum_j(A[i,j] * x[j])
kernel void matmul(
    const device FieldElement* A [[buffer(0)]],
    const device FieldElement* x [[buffer(1)]],
    device FieldElement* c [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= rows) return;

    FieldElement sum;
    for (int i = 0; i < 4; i++) sum.limbs[i] = 0;

    for (uint j = 0; j < cols; j++) {
        FieldElement a_ij = A[id * cols + j];
        FieldElement x_j = x[j];

        // TODO: Implement proper field multiplication and accumulation
        // For now, just accumulate first limbs (placeholder)
        sum.limbs[0] += a_ij.limbs[0] * x_j.limbs[0];
    }

    c[id] = sum;
}
