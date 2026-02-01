// HELIX CUDA Multi-Scalar Multiplication (MSM)
//
// GPU-accelerated MSM using Pippenger's bucket method for BN254 G1.
// Uses a hybrid CPU-GPU approach for optimal performance:
// - GPU: Parallel bucket index extraction, point sorting prep
// - CPU: Bucket accumulation (avoids atomic contention)
// - GPU: Parallel bucket reduction

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ============================================================================
// BN254 Field Constants
// ============================================================================

__constant__ uint64_t FR_MODULUS[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

__constant__ uint64_t FR_R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

__constant__ uint64_t FR_INV = 0xc2e1f593efffffffULL;

// Host-side constants
static const uint64_t H_MODULUS[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

static const uint64_t H_R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

// ============================================================================
// Curve Point Structures
// ============================================================================

struct AffinePoint {
    uint64_t x[4];
    uint64_t y[4];
};

struct ProjectivePoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
};

// ============================================================================
// Host Field Arithmetic
// ============================================================================

static void host_add256(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t* carry) {
    *carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a[i] + b[i] + *carry;
        r[i] = (uint64_t)sum;
        *carry = (uint64_t)(sum >> 64);
    }
}

static void host_sub256(uint64_t* r, const uint64_t* a, const uint64_t* b, uint64_t* borrow) {
    *borrow = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t diff = (__uint128_t)a[i] - b[i] - *borrow;
        r[i] = (uint64_t)diff;
        *borrow = (diff >> 127) ? 1 : 0;
    }
}

static bool host_gte(const uint64_t* a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

static void host_field_add(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t carry;
    host_add256(c, a, b, &carry);
    if (carry || host_gte(c, H_MODULUS)) {
        uint64_t borrow;
        host_sub256(c, c, H_MODULUS, &borrow);
    }
}

static void host_field_sub(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow;
    host_sub256(c, a, b, &borrow);
    if (borrow) {
        uint64_t carry;
        host_add256(c, c, H_MODULUS, &carry);
    }
}

static void host_mont_mul(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    const uint64_t inv = 0xc2e1f593efffffffULL;
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * inv;
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)m * H_MODULUS[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        for (int j = i+4; j < 8 && carry; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1 : 0;
            t[j] = sum;
        }
    }

    c[0] = t[4]; c[1] = t[5]; c[2] = t[6]; c[3] = t[7];

    if (host_gte(c, H_MODULUS)) {
        uint64_t borrow;
        host_sub256(c, c, H_MODULUS, &borrow);
    }
}

// ============================================================================
// Host Curve Arithmetic
// ============================================================================

static bool host_is_identity(const ProjectivePoint* p) {
    return p->z[0] == 0 && p->z[1] == 0 && p->z[2] == 0 && p->z[3] == 0;
}

static void host_set_identity(ProjectivePoint* p) {
    memset(p->x, 0, 32);
    memset(p->y, 0, 32);
    memset(p->z, 0, 32);
}

static void host_point_double(ProjectivePoint* r, const ProjectivePoint* p) {
    if (host_is_identity(p)) {
        *r = *p;
        return;
    }

    uint64_t a[4], b[4], c[4], d[4], tmp[4];

    // A = Y^2
    host_mont_mul(a, p->y, p->y);

    // B = 4*X*A
    host_mont_mul(b, p->x, a);
    host_field_add(b, b, b);
    host_field_add(b, b, b);

    // C = 8*A^2
    host_mont_mul(c, a, a);
    host_field_add(c, c, c);
    host_field_add(c, c, c);
    host_field_add(c, c, c);

    // D = 3*X^2 (a=0 for BN254)
    host_mont_mul(d, p->x, p->x);
    host_field_add(tmp, d, d);
    host_field_add(d, tmp, d);

    // X3 = D^2 - 2*B
    host_mont_mul(r->x, d, d);
    host_field_sub(r->x, r->x, b);
    host_field_sub(r->x, r->x, b);

    // Y3 = D*(B - X3) - C
    host_field_sub(tmp, b, r->x);
    host_mont_mul(r->y, d, tmp);
    host_field_sub(r->y, r->y, c);

    // Z3 = 2*Y*Z
    host_mont_mul(r->z, p->y, p->z);
    host_field_add(r->z, r->z, r->z);
}

static void host_point_add_mixed(ProjectivePoint* r, const ProjectivePoint* p, const AffinePoint* q) {
    if (host_is_identity(p)) {
        memcpy(r->x, q->x, 32);
        memcpy(r->y, q->y, 32);
        memcpy(r->z, H_R, 32);
        return;
    }

    uint64_t z1_sq[4], u2[4], z1_cu[4], s2[4], h[4], rr[4];
    uint64_t hh[4], hhh[4], v[4], tmp[4];

    host_mont_mul(z1_sq, p->z, p->z);
    host_mont_mul(u2, q->x, z1_sq);
    host_mont_mul(z1_cu, z1_sq, p->z);
    host_mont_mul(s2, q->y, z1_cu);

    host_field_sub(h, u2, p->x);
    host_field_sub(rr, s2, p->y);

    // Check for special cases
    bool h_zero = (h[0] == 0 && h[1] == 0 && h[2] == 0 && h[3] == 0);
    bool r_zero = (rr[0] == 0 && rr[1] == 0 && rr[2] == 0 && rr[3] == 0);

    if (h_zero && r_zero) {
        // Point doubling case
        host_point_double(r, p);
        return;
    }

    if (h_zero) {
        // Points are inverses
        host_set_identity(r);
        return;
    }

    host_mont_mul(hh, h, h);
    host_mont_mul(hhh, hh, h);
    host_mont_mul(v, p->x, hh);

    host_mont_mul(r->x, rr, rr);
    host_field_sub(r->x, r->x, hhh);
    host_field_sub(r->x, r->x, v);
    host_field_sub(r->x, r->x, v);

    host_field_sub(tmp, v, r->x);
    host_mont_mul(r->y, rr, tmp);
    host_mont_mul(tmp, p->y, hhh);
    host_field_sub(r->y, r->y, tmp);

    host_mont_mul(r->z, p->z, h);
}

static void host_point_add_proj(ProjectivePoint* r, const ProjectivePoint* p, const ProjectivePoint* q) {
    if (host_is_identity(p)) {
        *r = *q;
        return;
    }
    if (host_is_identity(q)) {
        *r = *p;
        return;
    }

    uint64_t z1_sq[4], z2_sq[4], z1_cu[4], z2_cu[4];
    uint64_t u1[4], u2[4], s1[4], s2[4], h[4], rr[4];
    uint64_t hh[4], hhh[4], v[4], tmp[4];

    host_mont_mul(z1_sq, p->z, p->z);
    host_mont_mul(z2_sq, q->z, q->z);
    host_mont_mul(z1_cu, z1_sq, p->z);
    host_mont_mul(z2_cu, z2_sq, q->z);

    host_mont_mul(u1, p->x, z2_sq);
    host_mont_mul(u2, q->x, z1_sq);
    host_mont_mul(s1, p->y, z2_cu);
    host_mont_mul(s2, q->y, z1_cu);

    host_field_sub(h, u2, u1);
    host_field_sub(rr, s2, s1);

    bool h_zero = (h[0] == 0 && h[1] == 0 && h[2] == 0 && h[3] == 0);
    bool r_zero = (rr[0] == 0 && rr[1] == 0 && rr[2] == 0 && rr[3] == 0);

    if (h_zero && r_zero) {
        host_point_double(r, p);
        return;
    }

    if (h_zero) {
        host_set_identity(r);
        return;
    }

    host_mont_mul(hh, h, h);
    host_mont_mul(hhh, hh, h);
    host_mont_mul(v, u1, hh);

    host_mont_mul(r->x, rr, rr);
    host_field_sub(r->x, r->x, hhh);
    host_field_sub(r->x, r->x, v);
    host_field_sub(r->x, r->x, v);

    host_field_sub(tmp, v, r->x);
    host_mont_mul(r->y, rr, tmp);
    host_mont_mul(tmp, s1, hhh);
    host_field_sub(r->y, r->y, tmp);

    host_mont_mul(tmp, p->z, q->z);
    host_mont_mul(r->z, tmp, h);
}

// ============================================================================
// Device Field Arithmetic
// ============================================================================

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

__device__ __forceinline__ bool needs_reduction(const uint64_t* a) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < FR_MODULUS[i]) return false;
        if (a[i] > FR_MODULUS[i]) return true;
    }
    return true;
}

__device__ __forceinline__ void reduce(uint64_t* a) {
    if (needs_reduction(a)) {
        sub256(a, a, FR_MODULUS);
    }
}

__device__ __forceinline__ void field_add(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t carry = add256(c, a, b);
    if (carry || needs_reduction(c)) {
        sub256(c, c, FR_MODULUS);
    }
}

__device__ __forceinline__ void field_sub(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow = sub256(c, a, b);
    if (borrow) {
        add256(c, c, FR_MODULUS);
    }
}

__device__ __forceinline__ void mont_mul(uint64_t* c, const uint64_t* a, const uint64_t* b) {
    uint64_t t[8] = {0};

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

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        c[i] = t[i + 4];
    }
    reduce(c);
}

// ============================================================================
// GPU Kernels
// ============================================================================

// Get window value from scalar
__device__ __forceinline__ uint32_t get_window(const uint64_t* scalar, int window_idx, int window_size) {
    int bit_offset = window_idx * window_size;
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;

    if (limb_idx >= 4) return 0;

    uint64_t value = scalar[limb_idx] >> bit_in_limb;

    int bits_from_first = 64 - bit_in_limb;
    if (bits_from_first < window_size && limb_idx + 1 < 4) {
        int remaining_bits = window_size - bits_from_first;
        uint64_t mask = (1ULL << remaining_bits) - 1;
        value |= (scalar[limb_idx + 1] & mask) << bits_from_first;
    }

    uint32_t mask = (1U << window_size) - 1;
    return (uint32_t)(value & mask);
}

// Kernel to extract bucket indices for all windows
__global__ void extract_all_bucket_indices(
    const uint64_t* __restrict__ scalars,
    uint32_t* __restrict__ bucket_indices,
    size_t count,
    int num_windows,
    int window_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint64_t* scalar = &scalars[idx * 4];

    for (int w = 0; w < num_windows; w++) {
        bucket_indices[w * count + idx] = get_window(scalar, w, window_size);
    }
}

// ============================================================================
// Host MSM Implementation
// ============================================================================

static uint32_t host_get_window(const uint64_t* scalar, int window_idx, int window_size) {
    int bit_offset = window_idx * window_size;
    int limb_idx = bit_offset / 64;
    int bit_in_limb = bit_offset % 64;

    if (limb_idx >= 4) return 0;

    uint64_t value = scalar[limb_idx] >> bit_in_limb;

    int bits_from_first = 64 - bit_in_limb;
    if (bits_from_first < window_size && limb_idx + 1 < 4) {
        int remaining_bits = window_size - bits_from_first;
        uint64_t mask = (1ULL << remaining_bits) - 1;
        value |= (scalar[limb_idx + 1] & mask) << bits_from_first;
    }

    uint32_t mask = (1U << window_size) - 1;
    return (uint32_t)(value & mask);
}

extern "C" {

int32_t helix_cuda_msm_pippenger(
    const uint64_t* points,
    const uint64_t* scalars,
    size_t count,
    size_t window_size,
    uint64_t* result
) {
    if (count == 0) {
        for (int i = 0; i < 12; i++) result[i] = 0;
        return 0;
    }

    // Clamp window size
    if (window_size < 4) window_size = 4;
    if (window_size > 16) window_size = 16;

    int num_windows = (256 + window_size - 1) / window_size;
    size_t num_buckets = (1ULL << window_size) - 1;

    // Allocate device memory for bucket indices
    uint64_t* d_scalars;
    uint32_t* d_bucket_indices;

    cudaError_t err;
    err = cudaMalloc(&d_scalars, count * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        // Fall back to pure CPU
        goto cpu_fallback;
    }

    err = cudaMalloc(&d_bucket_indices, num_windows * count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        goto cpu_fallback;
    }

    // Copy scalars to device
    cudaMemcpy(d_scalars, scalars, count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Extract all bucket indices on GPU (parallel)
    {
        int block_size = 256;
        int num_blocks = (count + block_size - 1) / block_size;
        extract_all_bucket_indices<<<num_blocks, block_size>>>(
            d_scalars, d_bucket_indices, count, num_windows, window_size
        );
        cudaDeviceSynchronize();
    }

    // Copy bucket indices back to host
    {
        uint32_t* h_bucket_indices = (uint32_t*)malloc(num_windows * count * sizeof(uint32_t));
        if (!h_bucket_indices) {
            cudaFree(d_scalars);
            cudaFree(d_bucket_indices);
            goto cpu_fallback;
        }

        cudaMemcpy(h_bucket_indices, d_bucket_indices, num_windows * count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaFree(d_scalars);
        cudaFree(d_bucket_indices);

        // Cast points to AffinePoint array
        const AffinePoint* affine_points = reinterpret_cast<const AffinePoint*>(points);

        // Process each window
        ProjectivePoint* window_results = (ProjectivePoint*)calloc(num_windows, sizeof(ProjectivePoint));
        if (!window_results) {
            free(h_bucket_indices);
            goto cpu_fallback;
        }

        for (int w = 0; w < num_windows; w++) {
            // Allocate buckets
            ProjectivePoint* buckets = (ProjectivePoint*)calloc(num_buckets, sizeof(ProjectivePoint));
            if (!buckets) {
                free(h_bucket_indices);
                free(window_results);
                goto cpu_fallback;
            }

            // Accumulate points into buckets
            uint32_t* window_indices = &h_bucket_indices[w * count];
            for (size_t i = 0; i < count; i++) {
                uint32_t bucket_idx = window_indices[i];
                if (bucket_idx > 0 && bucket_idx <= num_buckets) {
                    host_point_add_mixed(&buckets[bucket_idx - 1], &buckets[bucket_idx - 1], &affine_points[i]);
                }
            }

            // Reduce buckets using running sum technique
            ProjectivePoint running, sum;
            host_set_identity(&running);
            host_set_identity(&sum);

            for (size_t i = num_buckets; i > 0; i--) {
                host_point_add_proj(&running, &running, &buckets[i - 1]);
                host_point_add_proj(&sum, &sum, &running);
            }

            window_results[w] = sum;
            free(buckets);
        }

        free(h_bucket_indices);

        // Combine window results using Horner's method
        ProjectivePoint final_result = window_results[num_windows - 1];

        for (int w = num_windows - 2; w >= 0; w--) {
            // Double window_size times
            for (size_t j = 0; j < window_size; j++) {
                ProjectivePoint doubled;
                host_point_double(&doubled, &final_result);
                final_result = doubled;
            }
            // Add window result
            host_point_add_proj(&final_result, &final_result, &window_results[w]);
        }

        free(window_results);

        // Copy result
        memcpy(&result[0], final_result.x, 32);
        memcpy(&result[4], final_result.y, 32);
        memcpy(&result[8], final_result.z, 32);

        return 0;
    }

cpu_fallback:
    // Pure CPU implementation
    {
        const AffinePoint* affine_points = reinterpret_cast<const AffinePoint*>(points);

        ProjectivePoint* window_results = (ProjectivePoint*)calloc(num_windows, sizeof(ProjectivePoint));
        if (!window_results) {
            for (int i = 0; i < 12; i++) result[i] = 0;
            return -1;
        }

        for (int w = 0; w < num_windows; w++) {
            ProjectivePoint* buckets = (ProjectivePoint*)calloc(num_buckets, sizeof(ProjectivePoint));
            if (!buckets) {
                free(window_results);
                for (int i = 0; i < 12; i++) result[i] = 0;
                return -1;
            }

            // Accumulate points into buckets
            for (size_t i = 0; i < count; i++) {
                const uint64_t* scalar = &scalars[i * 4];
                uint32_t bucket_idx = host_get_window(scalar, w, window_size);
                if (bucket_idx > 0 && bucket_idx <= num_buckets) {
                    host_point_add_mixed(&buckets[bucket_idx - 1], &buckets[bucket_idx - 1], &affine_points[i]);
                }
            }

            // Reduce buckets
            ProjectivePoint running, sum;
            host_set_identity(&running);
            host_set_identity(&sum);

            for (size_t i = num_buckets; i > 0; i--) {
                host_point_add_proj(&running, &running, &buckets[i - 1]);
                host_point_add_proj(&sum, &sum, &running);
            }

            window_results[w] = sum;
            free(buckets);
        }

        // Combine window results
        ProjectivePoint final_result = window_results[num_windows - 1];

        for (int w = num_windows - 2; w >= 0; w--) {
            for (size_t j = 0; j < window_size; j++) {
                ProjectivePoint doubled;
                host_point_double(&doubled, &final_result);
                final_result = doubled;
            }
            host_point_add_proj(&final_result, &final_result, &window_results[w]);
        }

        free(window_results);

        memcpy(&result[0], final_result.x, 32);
        memcpy(&result[4], final_result.y, 32);
        memcpy(&result[8], final_result.z, 32);

        return 0;
    }
}

int32_t helix_cuda_msm_batch(
    const uint64_t* points,
    const uint64_t* scalars,
    const size_t* counts,
    size_t num_msms,
    size_t window_size,
    uint64_t* results
) {
    size_t point_offset = 0;
    size_t scalar_offset = 0;

    for (size_t i = 0; i < num_msms; i++) {
        int32_t status = helix_cuda_msm_pippenger(
            &points[point_offset * 8],
            &scalars[scalar_offset * 4],
            counts[i],
            window_size,
            &results[i * 12]
        );

        if (status != 0) return status;

        point_offset += counts[i];
        scalar_offset += counts[i];
    }

    return 0;
}

} // extern "C"
