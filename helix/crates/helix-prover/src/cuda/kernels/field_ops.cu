// HELIX CUDA Field Operations
//
// GPU-accelerated field arithmetic for BN254 scalar field.

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// BN254 Field Constants
// ============================================================================

// BN254 scalar field modulus: p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
__constant__ uint64_t FR_MODULUS[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

// Montgomery R = 2^256 mod p
__constant__ uint64_t FR_R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

// -p^{-1} mod 2^64
__constant__ uint64_t FR_INV = 0xc2e1f593efffffffULL;

// ============================================================================
// 64-bit Arithmetic Helpers
// ============================================================================

// 64x64 -> 128 bit multiplication
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
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

// Add 256-bit with carry
__device__ __forceinline__ uint64_t add256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]) ? 1 : 0;
        r[i] = sum;
    }
    return carry;
}

// Subtract 256-bit with borrow
__device__ __forceinline__ uint64_t sub256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i]) || (borrow && a[i] == b[i]) ? 1 : 0;
        r[i] = diff;
    }
    return borrow;
}

// Compare a >= modulus
__device__ __forceinline__ bool needs_reduction(const uint64_t* a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < FR_MODULUS[i]) return false;
        if (a[i] > FR_MODULUS[i]) return true;
    }
    return true; // Equal
}

// Reduce mod p
__device__ __forceinline__ void reduce(uint64_t* a) {
    if (needs_reduction(a)) {
        sub256(a, a, FR_MODULUS);
    }
}

// ============================================================================
// Field Operations
// ============================================================================

// Field addition kernel
__global__ void field_add_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t local_a[4], local_b[4], result[4];

    // Load inputs
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        local_a[i] = a[idx * 4 + i];
        local_b[i] = b[idx * 4 + i];
    }

    // Add
    uint64_t carry = add256(result, local_a, local_b);

    // Reduce if needed
    if (carry || needs_reduction(result)) {
        sub256(result, result, FR_MODULUS);
    }

    // Store result
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[idx * 4 + i] = result[i];
    }
}

// Field subtraction kernel
__global__ void field_sub_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t local_a[4], local_b[4], result[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        local_a[i] = a[idx * 4 + i];
        local_b[i] = b[idx * 4 + i];
    }

    uint64_t borrow = sub256(result, local_a, local_b);

    // Add modulus if borrowed
    if (borrow) {
        add256(result, result, FR_MODULUS);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[idx * 4 + i] = result[i];
    }
}

// Montgomery multiplication helper
__device__ __forceinline__ void mont_mul(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook multiplication
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(a[i], b[j], hi, lo);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) || (sum < lo) ? 1 : 0;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 4] = carry;
    }

    // Montgomery reduction
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * FR_INV;
        uint64_t carry = 0;

        #pragma unroll
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
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i] = t[i + 4];
    }
    reduce(c);
}

// Field multiplication kernel
__global__ void field_mul_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t local_a[4], local_b[4], result[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        local_a[i] = a[idx * 4 + i];
        local_b[i] = b[idx * 4 + i];
    }

    mont_mul(result, local_a, local_b);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[idx * 4 + i] = result[i];
    }
}

// ============================================================================
// External C Interface
// ============================================================================

extern "C" {

int32_t helix_cuda_field_add(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    size_t count
) {
    if (count == 0) return 0;

    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    field_add_kernel<<<num_blocks, block_size>>>(a, b, c, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_field_sub(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    size_t count
) {
    if (count == 0) return 0;

    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    field_sub_kernel<<<num_blocks, block_size>>>(a, b, c, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_field_mul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    size_t count
) {
    if (count == 0) return 0;

    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    field_mul_kernel<<<num_blocks, block_size>>>(a, b, c, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;

    err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : -1;
}

int32_t helix_cuda_field_batch_inv(
    const uint64_t* a,
    uint64_t* inv,
    size_t count
) {
    // Batch inversion is best done on CPU using Montgomery's trick
    // as it requires only one expensive inversion.
    // This is a placeholder that would need to implement the full algorithm.
    return -1; // Not implemented - fall back to CPU
}

} // extern "C"
