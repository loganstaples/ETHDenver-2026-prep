// HELIX CUDA Number-Theoretic Transform (NTT)
//
// GPU-accelerated NTT using Cooley-Tukey radix-2 algorithm with
// precomputed twiddle factors.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ============================================================================
// BN254 Field Constants
// ============================================================================

// BN254 scalar field modulus
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

// R^2 mod p (for Montgomery conversion)
__constant__ uint64_t FR_R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL
};

// -p^{-1} mod 2^64
__constant__ uint64_t FR_INV = 0xc2e1f593efffffffULL;

// Primitive 2^28-th root of unity in Montgomery form
__constant__ uint64_t ROOT_OF_UNITY[4] = {
    0x2a3c09f0a58a7e85ULL,
    0x5f4dda8b7c2c2acaULL,
    0x14a68b2f8e0c6c7bULL,
    0x2cf135e7506a7d2bULL
};

// ============================================================================
// Host-side Field Constants
// ============================================================================

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

static const uint64_t H_R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL
};

static const uint64_t H_ROOT[4] = {
    0x2a3c09f0a58a7e85ULL,
    0x5f4dda8b7c2c2acaULL,
    0x14a68b2f8e0c6c7bULL,
    0x2cf135e7506a7d2bULL
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

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    // Montgomery reduction
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

static void host_field_square(uint64_t* c, const uint64_t* a) {
    host_mont_mul(c, a, a);
}

static void host_field_pow(uint64_t* result, const uint64_t* base, const uint64_t* exp) {
    memcpy(result, H_R, 32);  // result = 1 in Montgomery
    uint64_t b[4];
    memcpy(b, base, 32);

    for (int i = 0; i < 4; i++) {
        uint64_t e = exp[i];
        for (int j = 0; j < 64; j++) {
            if (e & 1) {
                host_mont_mul(result, result, b);
            }
            host_field_square(b, b);
            e >>= 1;
        }
    }
}

static void host_field_inv(uint64_t* result, const uint64_t* a) {
    // a^{-1} = a^{p-2} mod p
    uint64_t exp[4] = {
        H_MODULUS[0] - 2,
        H_MODULUS[1],
        H_MODULUS[2],
        H_MODULUS[3]
    };
    host_field_pow(result, a, exp);
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
// NTT Kernels
// ============================================================================

// Cooley-Tukey butterfly: (a, b) -> (a + w*b, a - w*b)
__global__ void ntt_butterfly_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    size_t n,
    int stage,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t half_n = n >> 1;

    if (idx >= half_n) return;

    size_t block_size = 1ULL << (stage + 1);
    size_t half_block = block_size >> 1;

    size_t block_idx = idx / half_block;
    size_t idx_in_block = idx % half_block;

    size_t i = block_idx * block_size + idx_in_block;
    size_t j = i + half_block;

    size_t twiddle_idx = idx_in_block << (log_n - stage - 1);

    uint64_t a[4], b[4], w[4], wb[4];

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        a[k] = data[i * 4 + k];
        b[k] = data[j * 4 + k];
        w[k] = twiddles[twiddle_idx * 4 + k];
    }

    mont_mul(wb, w, b);

    uint64_t a_new[4], b_new[4];
    field_add(a_new, a, wb);
    field_sub(b_new, a, wb);

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        data[i * 4 + k] = a_new[k];
        data[j * 4 + k] = b_new[k];
    }
}

__global__ void bit_reverse_kernel(
    uint64_t* __restrict__ data,
    size_t n,
    int log_n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    size_t rev = 0;
    size_t x = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    if (idx < rev) {
        uint64_t temp[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            temp[k] = data[idx * 4 + k];
            data[idx * 4 + k] = data[rev * 4 + k];
            data[rev * 4 + k] = temp[k];
        }
    }
}

__global__ void ntt_scale_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t a[4], s[4], result[4];

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        a[k] = data[idx * 4 + k];
        s[k] = scale[k];
    }

    mont_mul(result, a, s);

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        data[idx * 4 + k] = result[k];
    }
}

// ============================================================================
// Twiddle Factor Management
// ============================================================================

static uint64_t* d_forward_twiddles = nullptr;
static uint64_t* d_inverse_twiddles = nullptr;
static uint64_t* d_size_inv = nullptr;
static size_t current_twiddle_size = 0;
static int current_log_n = 0;

static void compute_twiddles_host(uint64_t* forward, uint64_t* inverse, size_t n, int log_n) {
    // Get n-th root of unity from 2^28-th root
    uint64_t omega[4];
    memcpy(omega, H_ROOT, 32);

    // omega_n = omega^{2^{28-log_n}}
    for (int i = 0; i < 28 - log_n; i++) {
        host_field_square(omega, omega);
    }

    // Compute inverse of omega
    uint64_t omega_inv[4];
    host_field_inv(omega_inv, omega);

    // Compute forward twiddles: 1, omega, omega^2, ..., omega^{n-1}
    uint64_t current[4];
    memcpy(current, H_R, 32);  // 1 in Montgomery

    for (size_t i = 0; i < n; i++) {
        memcpy(&forward[i * 4], current, 32);
        host_mont_mul(current, current, omega);
    }

    // Compute inverse twiddles
    memcpy(current, H_R, 32);
    for (size_t i = 0; i < n; i++) {
        memcpy(&inverse[i * 4], current, 32);
        host_mont_mul(current, current, omega_inv);
    }
}

extern "C" {

int32_t helix_cuda_ntt_init_twiddles(size_t max_size) {
    if (max_size == 0 || (max_size & (max_size - 1)) != 0) {
        return -1;
    }

    int log_n = 0;
    for (size_t temp = max_size; temp > 1; temp >>= 1) log_n++;

    if (log_n > 28) return -1;  // BN254 only supports up to 2^28

    // Free existing twiddles
    if (d_forward_twiddles) cudaFree(d_forward_twiddles);
    if (d_inverse_twiddles) cudaFree(d_inverse_twiddles);
    if (d_size_inv) cudaFree(d_size_inv);

    // Allocate host memory for computation
    uint64_t* h_forward = (uint64_t*)malloc(max_size * 4 * sizeof(uint64_t));
    uint64_t* h_inverse = (uint64_t*)malloc(max_size * 4 * sizeof(uint64_t));

    if (!h_forward || !h_inverse) {
        free(h_forward);
        free(h_inverse);
        return -1;
    }

    // Compute twiddles on host
    compute_twiddles_host(h_forward, h_inverse, max_size, log_n);

    // Allocate device memory
    cudaError_t err;
    err = cudaMalloc(&d_forward_twiddles, max_size * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        free(h_forward);
        free(h_inverse);
        return -1;
    }

    err = cudaMalloc(&d_inverse_twiddles, max_size * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_forward_twiddles);
        d_forward_twiddles = nullptr;
        free(h_forward);
        free(h_inverse);
        return -1;
    }

    err = cudaMalloc(&d_size_inv, 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_forward_twiddles);
        cudaFree(d_inverse_twiddles);
        d_forward_twiddles = nullptr;
        d_inverse_twiddles = nullptr;
        free(h_forward);
        free(h_inverse);
        return -1;
    }

    // Copy to device
    cudaMemcpy(d_forward_twiddles, h_forward, max_size * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverse_twiddles, h_inverse, max_size * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Compute n^{-1} for INTT scaling
    uint64_t n_mont[4] = {max_size, 0, 0, 0};
    uint64_t temp[4];
    host_mont_mul(temp, n_mont, H_R2);  // Convert to Montgomery
    uint64_t n_inv[4];
    host_field_inv(n_inv, temp);
    cudaMemcpy(d_size_inv, n_inv, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    free(h_forward);
    free(h_inverse);

    current_twiddle_size = max_size;
    current_log_n = log_n;

    return 0;
}

int32_t helix_cuda_ntt_cleanup() {
    if (d_forward_twiddles) {
        cudaFree(d_forward_twiddles);
        d_forward_twiddles = nullptr;
    }
    if (d_inverse_twiddles) {
        cudaFree(d_inverse_twiddles);
        d_inverse_twiddles = nullptr;
    }
    if (d_size_inv) {
        cudaFree(d_size_inv);
        d_size_inv = nullptr;
    }
    current_twiddle_size = 0;
    current_log_n = 0;
    return 0;
}

int32_t helix_cuda_ntt_forward(uint64_t* data, size_t count) {
    if (count == 0 || (count & (count - 1)) != 0) {
        return -1;
    }

    int log_n = 0;
    for (size_t temp = count; temp > 1; temp >>= 1) log_n++;

    // Auto-initialize twiddles if needed
    if (d_forward_twiddles == nullptr || count > current_twiddle_size) {
        size_t init_size = count;
        // Round up to reasonable size
        if (init_size < 1024) init_size = 1024;
        if (init_size < 65536) init_size = 65536;
        int32_t status = helix_cuda_ntt_init_twiddles(init_size);
        if (status != 0) return status;
    }

    // Allocate device memory
    uint64_t* d_data;
    cudaError_t err = cudaMalloc(&d_data, count * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return -1;

    cudaMemcpy(d_data, data, count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    // Bit-reverse permutation
    bit_reverse_kernel<<<num_blocks, block_size>>>(d_data, count, log_n);
    cudaDeviceSynchronize();

    // Execute all NTT butterfly stages
    for (int stage = 0; stage < log_n; stage++) {
        size_t half_n = count >> 1;
        num_blocks = (half_n + block_size - 1) / block_size;
        ntt_butterfly_kernel<<<num_blocks, block_size>>>(
            d_data, d_forward_twiddles, count, stage, log_n
        );
        cudaDeviceSynchronize();
    }

    // Copy result back
    cudaMemcpy(data, d_data, count * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    return 0;
}

int32_t helix_cuda_ntt_inverse(uint64_t* data, size_t count) {
    if (count == 0 || (count & (count - 1)) != 0) {
        return -1;
    }

    int log_n = 0;
    for (size_t temp = count; temp > 1; temp >>= 1) log_n++;

    // Auto-initialize twiddles if needed
    if (d_inverse_twiddles == nullptr || count > current_twiddle_size) {
        size_t init_size = count;
        if (init_size < 1024) init_size = 1024;
        if (init_size < 65536) init_size = 65536;
        int32_t status = helix_cuda_ntt_init_twiddles(init_size);
        if (status != 0) return status;
    }

    uint64_t* d_data;
    cudaError_t err = cudaMalloc(&d_data, count * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return -1;

    cudaMemcpy(d_data, data, count * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;

    // Bit-reverse permutation
    bit_reverse_kernel<<<num_blocks, block_size>>>(d_data, count, log_n);
    cudaDeviceSynchronize();

    // Execute all INTT butterfly stages
    for (int stage = 0; stage < log_n; stage++) {
        size_t half_n = count >> 1;
        num_blocks = (half_n + block_size - 1) / block_size;
        ntt_butterfly_kernel<<<num_blocks, block_size>>>(
            d_data, d_inverse_twiddles, count, stage, log_n
        );
        cudaDeviceSynchronize();
    }

    // Scale by n^{-1}
    // Recompute n_inv for this specific size
    uint64_t n_mont[4] = {count, 0, 0, 0};
    uint64_t temp[4];
    host_mont_mul(temp, n_mont, H_R2);
    uint64_t n_inv[4];
    host_field_inv(n_inv, temp);

    uint64_t* d_scale;
    cudaMalloc(&d_scale, 4 * sizeof(uint64_t));
    cudaMemcpy(d_scale, n_inv, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    num_blocks = (count + block_size - 1) / block_size;
    ntt_scale_kernel<<<num_blocks, block_size>>>(d_data, d_scale, count);
    cudaDeviceSynchronize();

    cudaFree(d_scale);

    cudaMemcpy(data, d_data, count * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    return 0;
}

int32_t helix_cuda_ntt_batch(
    uint64_t* data,
    size_t count,
    size_t batch_size,
    int32_t inverse
) {
    for (size_t i = 0; i < batch_size; i++) {
        uint64_t* ntt_data = &data[i * count * 4];
        int32_t status;

        if (inverse) {
            status = helix_cuda_ntt_inverse(ntt_data, count);
        } else {
            status = helix_cuda_ntt_forward(ntt_data, count);
        }

        if (status != 0) return status;
    }

    return 0;
}

} // extern "C"
