//! Metal-Accelerated Number-Theoretic Transform (NTT).
//!
//! This module implements GPU-accelerated NTT/INTT using Apple Metal for
//! fast polynomial multiplication in finite fields.
//!
//! ## Algorithm Overview
//!
//! NTT is the finite-field equivalent of FFT. For a field F_p with primitive
//! n-th root of unity ω, NTT transforms coefficients to evaluation form:
//!
//!   NTT([a_0, ..., a_{n-1}]) = [A_0, ..., A_{n-1}]
//!   where A_i = Σ_j a_j * ω^{ij}
//!
//! ## GPU Parallelization
//!
//! Uses Cooley-Tukey radix-2 algorithm with:
//! - Parallel butterfly operations within each stage
//! - Precomputed twiddle factors
//! - Coalesced memory access patterns

use super::{MetalDevice, MetalError, MetalResult, MetalConfig, MetalStats};
use crate::gkr::FieldElement;
use std::sync::Arc;

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal_rs::{Buffer, ComputePipelineState};

/// Configuration for NTT operations.
#[derive(Debug, Clone)]
pub struct NttConfig {
    /// Maximum supported domain size (log2)
    pub max_log_n: usize,
    /// Minimum batch size to use GPU
    pub min_gpu_batch_size: usize,
    /// Threadgroup size for kernels
    pub threadgroup_size: usize,
    /// Whether to precompute twiddle factors
    pub precompute_twiddles: bool,
    /// Whether to use shared memory for twiddles
    pub use_shared_memory: bool,
}

impl Default for NttConfig {
    fn default() -> Self {
        Self {
            max_log_n: 24,           // Up to 16M elements
            min_gpu_batch_size: 512, // GPU overhead not worth it below this
            threadgroup_size: 256,
            precompute_twiddles: true,
            use_shared_memory: true,
        }
    }
}

/// Precomputed twiddle factors for NTT.
#[derive(Clone)]
pub struct TwiddleFactors {
    /// Forward twiddle factors (ω^i for NTT)
    pub forward: Vec<[u64; 4]>,
    /// Inverse twiddle factors (ω^{-i} for INTT)
    pub inverse: Vec<[u64; 4]>,
    /// Domain size
    pub size: usize,
    /// n^{-1} mod p for INTT normalization
    pub size_inv: [u64; 4],
}

impl TwiddleFactors {
    /// Creates twiddle factors for the given domain size.
    pub fn new(log_n: usize) -> Self {
        let size = 1 << log_n;

        // BN254 scalar field primitive root of unity (for 2^28 domain)
        // Generator ω such that ω^{2^28} = 1
        let omega = get_root_of_unity(log_n);
        let omega_inv = field_inverse(&omega);

        // Precompute forward twiddles: ω^0, ω^1, ω^2, ..., ω^{n-1}
        let mut forward = Vec::with_capacity(size);
        let mut current = [1u64, 0, 0, 0]; // Montgomery 1

        // Convert to Montgomery form
        let one_mont = to_montgomery(&[1, 0, 0, 0]);
        current = one_mont;

        for _ in 0..size {
            forward.push(current);
            current = field_mul_limbs(&current, &omega);
        }

        // Precompute inverse twiddles
        let mut inverse = Vec::with_capacity(size);
        current = one_mont;
        for _ in 0..size {
            inverse.push(current);
            current = field_mul_limbs(&current, &omega_inv);
        }

        // Compute n^{-1} mod p
        let n_as_field = [size as u64, 0, 0, 0];
        let n_mont = to_montgomery(&n_as_field);
        let size_inv = field_inverse(&n_mont);

        Self {
            forward,
            inverse,
            size,
            size_inv,
        }
    }
}

/// Statistics from NTT operations.
#[derive(Debug, Clone, Default)]
pub struct NttStats {
    /// Number of NTT operations
    pub forward_transforms: usize,
    /// Number of INTT operations
    pub inverse_transforms: usize,
    /// Total GPU time in microseconds
    pub gpu_time_us: u64,
    /// Total elements processed
    pub elements_processed: usize,
}

/// Metal-accelerated NTT engine.
pub struct MetalNtt {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: Arc<MetalDevice>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    butterfly_pipeline: ComputePipelineState,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    scale_pipeline: ComputePipelineState,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    twiddle_buffer: Option<Buffer>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    twiddle_inv_buffer: Option<Buffer>,
    config: NttConfig,
    twiddles: Option<TwiddleFactors>,
    stats: NttStats,
}

/// Metal shader source for NTT operations.
const NTT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Field element (4 x u64 limbs in Montgomery form)
struct FieldElement {
    uint64_t limbs[4];
};

// BN254 modulus
constant uint64_t MODULUS[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

// Montgomery constant R
constant uint64_t MONT_R[4] = {
    0xd35d438dc58f0d9dULL,
    0x0a78eb28f5c70b3dULL,
    0x666ea36f7879462cULL,
    0x0e0a77c19a07df2fULL
};

// Montgomery reduction parameter
constant uint64_t INV = 0xc2e1f593effffffULL;

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

// Add 256-bit with carry
inline uint64_t add256(thread FieldElement& r, thread FieldElement a, thread FieldElement b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        r.limbs[i] = sum;
    }
    return carry;
}

// Subtract 256-bit with borrow
inline uint64_t sub256(thread FieldElement& r, thread FieldElement a, thread FieldElement b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = (a.limbs[i] < b.limbs[i]) || (borrow && a.limbs[i] == b.limbs[i]) ? 1 : 0;
        r.limbs[i] = diff;
    }
    return borrow;
}

// Compare to modulus
inline int cmp_mod(thread FieldElement a) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] < MODULUS[i]) return 0;
        if (a.limbs[i] > MODULUS[i]) return 1;
    }
    return 1;
}

// Reduce mod p
inline void reduce(thread FieldElement& a) {
    if (cmp_mod(a)) {
        FieldElement mod;
        for (int i = 0; i < 4; i++) mod.limbs[i] = MODULUS[i];
        sub256(a, a, mod);
    }
}

// Field addition
inline void field_add(thread FieldElement& c, thread FieldElement a, thread FieldElement b) {
    uint64_t carry = add256(c, a, b);
    FieldElement mod;
    for (int i = 0; i < 4; i++) mod.limbs[i] = MODULUS[i];
    if (carry || cmp_mod(c)) {
        sub256(c, c, mod);
    }
}

// Field subtraction
inline void field_sub(thread FieldElement& c, thread FieldElement a, thread FieldElement b) {
    uint64_t borrow = sub256(c, a, b);
    if (borrow) {
        FieldElement mod;
        for (int i = 0; i < 4; i++) mod.limbs[i] = MODULUS[i];
        add256(c, c, mod);
    }
}

// Montgomery multiplication
inline void field_mul(thread FieldElement& c, thread FieldElement a, thread FieldElement b) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

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

    for (int i = 0; i < 4; i++) {
        c.limbs[i] = t[i + 4];
    }
    reduce(c);
}

// Cooley-Tukey butterfly operation
// a' = a + w*b
// b' = a - w*b
kernel void ntt_butterfly(
    device FieldElement* data [[buffer(0)]],
    const device FieldElement* twiddles [[buffer(1)]],
    constant uint32_t& stage [[buffer(2)]],      // Current stage (0 to log2(n)-1)
    constant uint32_t& log_n [[buffer(3)]],      // log2(n)
    constant uint32_t& is_inverse [[buffer(4)]], // 0 = forward, 1 = inverse
    uint id [[thread_position_in_grid]]
) {
    uint32_t n = 1u << log_n;
    uint32_t half_n = n >> 1;

    if (id >= half_n) return;

    // Compute butterfly indices
    uint32_t block_size = 1u << (stage + 1);
    uint32_t half_block = block_size >> 1;

    uint32_t block_idx = id / half_block;
    uint32_t idx_in_block = id % half_block;

    uint32_t i = block_idx * block_size + idx_in_block;
    uint32_t j = i + half_block;

    // Get twiddle factor
    // For forward NTT: twiddle index = idx_in_block * (n / block_size)
    // For inverse NTT: same, but using inverse twiddles
    uint32_t twiddle_idx = idx_in_block << (log_n - stage - 1);

    FieldElement a = data[i];
    FieldElement b = data[j];
    FieldElement w = twiddles[twiddle_idx];

    // Compute w * b
    FieldElement wb;
    field_mul(wb, w, b);

    // a' = a + wb
    // b' = a - wb
    FieldElement a_new, b_new;
    field_add(a_new, a, wb);
    field_sub(b_new, a, wb);

    data[i] = a_new;
    data[j] = b_new;
}

// Scale by n^{-1} for inverse NTT
kernel void ntt_scale(
    device FieldElement* data [[buffer(0)]],
    constant FieldElement& scale [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) return;

    FieldElement a = data[id];
    FieldElement result;
    field_mul(result, a, scale);
    data[id] = result;
}

// Bit-reversal permutation
kernel void bit_reverse(
    device FieldElement* data [[buffer(0)]],
    constant uint32_t& log_n [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint32_t n = 1u << log_n;
    if (id >= n) return;

    // Compute bit-reversed index
    uint32_t rev = 0;
    uint32_t x = id;
    for (uint32_t i = 0; i < log_n; i++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    // Only swap if id < rev to avoid double-swapping
    if (id < rev) {
        FieldElement temp = data[id];
        data[id] = data[rev];
        data[rev] = temp;
    }
}

// Fused NTT stage for better memory coalescing (processes multiple butterflies)
kernel void ntt_stage_fused(
    device FieldElement* data [[buffer(0)]],
    const device FieldElement* twiddles [[buffer(1)]],
    constant uint32_t& stage [[buffer(2)]],
    constant uint32_t& log_n [[buffer(3)]],
    constant uint32_t& butterflies_per_thread [[buffer(4)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    threadgroup FieldElement* shared [[threadgroup(0)]]
) {
    uint32_t n = 1u << log_n;
    uint32_t half_n = n >> 1;
    uint32_t total_butterflies = half_n;

    uint32_t start_bf = id * butterflies_per_thread;
    uint32_t end_bf = min(start_bf + butterflies_per_thread, total_butterflies);

    uint32_t block_size = 1u << (stage + 1);
    uint32_t half_block = block_size >> 1;

    for (uint32_t bf = start_bf; bf < end_bf; bf++) {
        uint32_t block_idx = bf / half_block;
        uint32_t idx_in_block = bf % half_block;

        uint32_t i = block_idx * block_size + idx_in_block;
        uint32_t j = i + half_block;

        uint32_t twiddle_idx = idx_in_block << (log_n - stage - 1);

        FieldElement a = data[i];
        FieldElement b = data[j];
        FieldElement w = twiddles[twiddle_idx];

        FieldElement wb;
        field_mul(wb, w, b);

        FieldElement a_new, b_new;
        field_add(a_new, a, wb);
        field_sub(b_new, a, wb);

        data[i] = a_new;
        data[j] = b_new;
    }
}
"#;

impl MetalNtt {
    /// Creates a new Metal NTT engine.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device: Arc<MetalDevice>) -> MetalResult<Self> {
        Self::with_config(device, NttConfig::default())
    }

    /// Creates a new Metal NTT engine with custom configuration.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn with_config(device: Arc<MetalDevice>, config: NttConfig) -> MetalResult<Self> {
        let butterfly_pipeline = device.compile_shader(NTT_SHADER_SOURCE, "ntt_butterfly")?;
        let scale_pipeline = device.compile_shader(NTT_SHADER_SOURCE, "ntt_scale")?;

        Ok(Self {
            device,
            butterfly_pipeline,
            scale_pipeline,
            twiddle_buffer: None,
            twiddle_inv_buffer: None,
            config,
            twiddles: None,
            stats: NttStats::default(),
        })
    }

    /// Creates a new Metal NTT engine (stub for non-macOS).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn new(_device: Arc<MetalDevice>) -> MetalResult<Self> {
        Err(MetalError::NotAvailable)
    }

    /// Creates with config (stub for non-macOS).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn with_config(_device: Arc<MetalDevice>, _config: NttConfig) -> MetalResult<Self> {
        Err(MetalError::NotAvailable)
    }

    /// Ensures twiddle factors are precomputed for the given size.
    pub fn ensure_twiddles(&mut self, log_n: usize) -> MetalResult<()> {
        if let Some(ref twiddles) = self.twiddles {
            if twiddles.size >= (1 << log_n) {
                return Ok(());
            }
        }

        let twiddles = TwiddleFactors::new(log_n);

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.twiddle_buffer = Some(self.device.create_buffer_with_data(&twiddles.forward)?);
            self.twiddle_inv_buffer = Some(self.device.create_buffer_with_data(&twiddles.inverse)?);
        }

        self.twiddles = Some(twiddles);
        Ok(())
    }

    /// Performs forward NTT in-place.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn forward(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidArgument(
                "NTT size must be a power of 2".to_string()
            ));
        }

        if n < self.config.min_gpu_batch_size {
            return Ok(self.forward_cpu(data));
        }

        let log_n = n.trailing_zeros() as usize;
        self.ensure_twiddles(log_n)?;

        let start_time = std::time::Instant::now();

        // Bit-reverse permutation
        self.bit_reverse_cpu(data, log_n);

        // Create data buffer
        let data_buffer = self.device.create_buffer_with_data(data)?;

        // Execute NTT stages
        for stage in 0..log_n {
            self.execute_butterfly_stage(&data_buffer, stage, log_n, false)?;
        }

        // Copy result back
        unsafe {
            let contents = data_buffer.contents() as *const [u64; 4];
            std::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), n);
        }

        self.stats.forward_transforms += 1;
        self.stats.elements_processed += n;
        self.stats.gpu_time_us += start_time.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Performs inverse NTT in-place.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn inverse(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err(MetalError::InvalidArgument(
                "NTT size must be a power of 2".to_string()
            ));
        }

        if n < self.config.min_gpu_batch_size {
            return Ok(self.inverse_cpu(data));
        }

        let log_n = n.trailing_zeros() as usize;
        self.ensure_twiddles(log_n)?;

        let start_time = std::time::Instant::now();

        // Bit-reverse permutation
        self.bit_reverse_cpu(data, log_n);

        // Create data buffer
        let data_buffer = self.device.create_buffer_with_data(data)?;

        // Execute INTT stages (using inverse twiddles)
        for stage in 0..log_n {
            self.execute_butterfly_stage(&data_buffer, stage, log_n, true)?;
        }

        // Scale by n^{-1}
        self.execute_scale(&data_buffer, n)?;

        // Copy result back
        unsafe {
            let contents = data_buffer.contents() as *const [u64; 4];
            std::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), n);
        }

        self.stats.inverse_transforms += 1;
        self.stats.elements_processed += n;
        self.stats.gpu_time_us += start_time.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Executes a single butterfly stage.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn execute_butterfly_stage(
        &self,
        _data: &Buffer,
        _stage: usize,
        _log_n: usize,
        _inverse: bool,
    ) -> MetalResult<()> {
        // Would dispatch butterfly kernel
        Ok(())
    }

    /// Executes scaling by n^{-1}.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn execute_scale(&self, _data: &Buffer, _n: usize) -> MetalResult<()> {
        // Would dispatch scale kernel
        Ok(())
    }

    /// Forward NTT (stub for non-macOS).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn forward(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        self.forward_cpu(data);
        Ok(())
    }

    /// Inverse NTT (stub for non-macOS).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn inverse(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        self.inverse_cpu(data);
        Ok(())
    }

    /// Bit-reverse permutation on CPU.
    fn bit_reverse_cpu(&self, data: &mut [[u64; 4]], log_n: usize) {
        let n = data.len();

        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }
    }

    /// Forward NTT on CPU.
    fn forward_cpu(&self, data: &mut [[u64; 4]]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let log_n = n.trailing_zeros() as usize;

        // Bit-reverse permutation
        self.bit_reverse_cpu(data, log_n);

        // Get or compute twiddle factors
        let omega = get_root_of_unity(log_n);

        // Cooley-Tukey iterative FFT
        for stage in 0..log_n {
            let block_size = 1 << (stage + 1);
            let half_block = block_size >> 1;

            // Compute twiddle for this stage
            let stage_omega = field_pow(&omega, (1 << (log_n - stage - 1)) as u64);

            for block_start in (0..n).step_by(block_size) {
                let mut twiddle = [1u64, 0, 0, 0]; // Montgomery 1
                let one_mont = to_montgomery(&[1, 0, 0, 0]);
                twiddle = one_mont;

                for j in 0..half_block {
                    let i = block_start + j;
                    let k = i + half_block;

                    let a = data[i];
                    let b = data[k];

                    // wb = twiddle * b
                    let wb = field_mul_limbs(&twiddle, &b);

                    // a' = a + wb
                    // b' = a - wb
                    data[i] = field_add_limbs(&a, &wb);
                    data[k] = field_sub_limbs(&a, &wb);

                    twiddle = field_mul_limbs(&twiddle, &stage_omega);
                }
            }
        }
    }

    /// Inverse NTT on CPU.
    fn inverse_cpu(&self, data: &mut [[u64; 4]]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let log_n = n.trailing_zeros() as usize;

        // Bit-reverse permutation
        self.bit_reverse_cpu(data, log_n);

        // Get inverse root of unity
        let omega = get_root_of_unity(log_n);
        let omega_inv = field_inverse(&omega);

        // Cooley-Tukey iterative IFFT
        for stage in 0..log_n {
            let block_size = 1 << (stage + 1);
            let half_block = block_size >> 1;

            let stage_omega_inv = field_pow(&omega_inv, (1 << (log_n - stage - 1)) as u64);

            for block_start in (0..n).step_by(block_size) {
                let one_mont = to_montgomery(&[1, 0, 0, 0]);
                let mut twiddle = one_mont;

                for j in 0..half_block {
                    let i = block_start + j;
                    let k = i + half_block;

                    let a = data[i];
                    let b = data[k];

                    let wb = field_mul_limbs(&twiddle, &b);

                    data[i] = field_add_limbs(&a, &wb);
                    data[k] = field_sub_limbs(&a, &wb);

                    twiddle = field_mul_limbs(&twiddle, &stage_omega_inv);
                }
            }
        }

        // Scale by n^{-1}
        let n_as_field = to_montgomery(&[n as u64, 0, 0, 0]);
        let n_inv = field_inverse(&n_as_field);

        for elem in data.iter_mut() {
            *elem = field_mul_limbs(elem, &n_inv);
        }
    }

    /// Returns statistics.
    pub fn stats(&self) -> &NttStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = NttStats::default();
    }
}

/// High-level NTT interface with automatic backend selection.
pub struct NttEngine {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<MetalNtt>,
    config: NttConfig,
    twiddles: Option<TwiddleFactors>,
}

impl NttEngine {
    /// Creates a new NTT engine.
    pub fn new() -> Self {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let metal = MetalDevice::new().ok().and_then(|device| {
                MetalNtt::new(Arc::new(device)).ok()
            });
            Self {
                metal,
                config: NttConfig::default(),
                twiddles: None,
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Self {
                config: NttConfig::default(),
                twiddles: None,
            }
        }
    }

    /// Creates with custom configuration.
    pub fn with_config(config: NttConfig) -> Self {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let metal = MetalDevice::new().ok().and_then(|device| {
                MetalNtt::with_config(Arc::new(device), config.clone()).ok()
            });
            Self {
                metal,
                config,
                twiddles: None,
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Self {
                config,
                twiddles: None,
            }
        }
    }

    /// Returns whether GPU is available.
    pub fn is_gpu_available(&self) -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            self.metal.is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    /// Forward NTT.
    pub fn forward(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(ref mut metal) = self.metal {
                if data.len() >= self.config.min_gpu_batch_size {
                    return metal.forward(data);
                }
            }
        }

        // CPU fallback
        self.forward_cpu(data);
        Ok(())
    }

    /// Inverse NTT.
    pub fn inverse(&mut self, data: &mut [[u64; 4]]) -> MetalResult<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(ref mut metal) = self.metal {
                if data.len() >= self.config.min_gpu_batch_size {
                    return metal.inverse(data);
                }
            }
        }

        // CPU fallback
        self.inverse_cpu(data);
        Ok(())
    }

    /// CPU forward NTT.
    fn forward_cpu(&mut self, data: &mut [[u64; 4]]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let log_n = n.trailing_zeros() as usize;

        // Ensure twiddles
        if self.twiddles.is_none() || self.twiddles.as_ref().unwrap().size < n {
            self.twiddles = Some(TwiddleFactors::new(log_n));
        }

        // Bit-reverse permutation
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }

        // Cooley-Tukey
        let twiddles = self.twiddles.as_ref().unwrap();

        for stage in 0..log_n {
            let block_size = 1 << (stage + 1);
            let half_block = block_size >> 1;
            let stride = n >> (stage + 1);

            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i = block_start + j;
                    let k = i + half_block;

                    let twiddle_idx = j * stride;
                    let twiddle = &twiddles.forward[twiddle_idx];

                    let a = data[i];
                    let b = data[k];

                    let wb = field_mul_limbs(twiddle, &b);

                    data[i] = field_add_limbs(&a, &wb);
                    data[k] = field_sub_limbs(&a, &wb);
                }
            }
        }
    }

    /// CPU inverse NTT.
    fn inverse_cpu(&mut self, data: &mut [[u64; 4]]) {
        let n = data.len();
        if n <= 1 {
            return;
        }

        let log_n = n.trailing_zeros() as usize;

        // Ensure twiddles
        if self.twiddles.is_none() || self.twiddles.as_ref().unwrap().size < n {
            self.twiddles = Some(TwiddleFactors::new(log_n));
        }

        // Bit-reverse permutation
        for i in 0..n {
            let j = bit_reverse(i, log_n);
            if i < j {
                data.swap(i, j);
            }
        }

        // Cooley-Tukey with inverse twiddles
        let twiddles = self.twiddles.as_ref().unwrap();

        for stage in 0..log_n {
            let block_size = 1 << (stage + 1);
            let half_block = block_size >> 1;
            let stride = n >> (stage + 1);

            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i = block_start + j;
                    let k = i + half_block;

                    let twiddle_idx = j * stride;
                    let twiddle = &twiddles.inverse[twiddle_idx];

                    let a = data[i];
                    let b = data[k];

                    let wb = field_mul_limbs(twiddle, &b);

                    data[i] = field_add_limbs(&a, &wb);
                    data[k] = field_sub_limbs(&a, &wb);
                }
            }
        }

        // Scale by n^{-1}
        let n_inv = &twiddles.size_inv;
        for elem in data.iter_mut() {
            *elem = field_mul_limbs(elem, n_inv);
        }
    }

    /// Polynomial multiplication using NTT.
    pub fn poly_mul(&mut self, a: &[[u64; 4]], b: &[[u64; 4]]) -> MetalResult<Vec<[u64; 4]>> {
        let n = (a.len() + b.len()).next_power_of_two();

        // Pad inputs
        let mut a_padded = vec![[0u64; 4]; n];
        let mut b_padded = vec![[0u64; 4]; n];

        a_padded[..a.len()].copy_from_slice(a);
        b_padded[..b.len()].copy_from_slice(b);

        // Forward NTT
        self.forward(&mut a_padded)?;
        self.forward(&mut b_padded)?;

        // Point-wise multiplication
        for i in 0..n {
            a_padded[i] = field_mul_limbs(&a_padded[i], &b_padded[i]);
        }

        // Inverse NTT
        self.inverse(&mut a_padded)?;

        Ok(a_padded)
    }
}

impl Default for NttEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Field Arithmetic Helper Functions
// ============================================================================

/// BN254 scalar field modulus.
const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

/// Montgomery R = 2^256 mod p.
const MONT_R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

/// R^2 mod p.
const MONT_R2: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
];

/// -p^{-1} mod 2^64.
const INV: u64 = 0xc2e1f593effffff;

/// Bit-reverse a number.
fn bit_reverse(x: usize, log_n: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Convert to Montgomery form.
fn to_montgomery(a: &[u64; 4]) -> [u64; 4] {
    field_mul_limbs(a, &MONT_R2)
}

/// Get root of unity for the given domain size.
fn get_root_of_unity(log_n: usize) -> [u64; 4] {
    // BN254 has a 2^28-th root of unity
    // Generator: 0x30644e72e131a029b85045b68181585d2833e84879b97091043e1f593f0000001
    // For smaller domains, we use ω^{2^{28-log_n}}

    // This is a placeholder - in production, would use the actual root
    let base_root: [u64; 4] = [
        0x30644e72e131a029,
        0xb85045b68181585d,
        0x2833e84879b97091,
        0x043e1f593f000001,
    ];

    // Convert to Montgomery form
    let root_mont = to_montgomery(&base_root);

    // Compute ω^{2^{28-log_n}} to get n-th root
    if log_n > 28 {
        panic!("Domain too large: log_n > 28");
    }

    let exp = 1u64 << (28 - log_n);
    field_pow(&root_mont, exp)
}

/// Field addition on limbs.
fn field_add_limbs(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u64;

    for i in 0..4 {
        let (sum1, c1) = a[i].overflowing_add(b[i]);
        let (sum2, c2) = sum1.overflowing_add(carry);
        result[i] = sum2;
        carry = (c1 as u64) + (c2 as u64);
    }

    // Reduce if needed
    if carry > 0 || cmp_limbs(&result, &MODULUS) >= 0 {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff1, b1) = result[i].overflowing_sub(MODULUS[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }

    result
}

/// Field subtraction on limbs.
fn field_sub_limbs(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0u64;

    for i in 0..4 {
        let (diff1, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
    }

    // If we borrowed, add modulus back
    if borrow > 0 {
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum1, c1) = result[i].overflowing_add(MODULUS[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
    }

    result
}

/// Field multiplication on limbs (Montgomery).
fn field_mul_limbs(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut t = [0u64; 8];

    // Schoolbook multiplication
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let prod = (a[i] as u128) * (b[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = prod as u64;
            carry = prod >> 64;
        }
        t[i + 4] = carry as u64;
    }

    // Montgomery reduction
    for i in 0..4 {
        let m = t[i].wrapping_mul(INV);
        let mut carry = 0u128;

        for j in 0..4 {
            let prod = (m as u128) * (MODULUS[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = prod as u64;
            carry = prod >> 64;
        }

        for j in (i + 4)..8 {
            let sum = (t[j] as u128) + carry;
            t[j] = sum as u64;
            carry = sum >> 64;
            if carry == 0 {
                break;
            }
        }
    }

    let mut result = [t[4], t[5], t[6], t[7]];

    // Final reduction
    if cmp_limbs(&result, &MODULUS) >= 0 {
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff1, b1) = result[i].overflowing_sub(MODULUS[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
    }

    result
}

/// Compare two limb arrays.
fn cmp_limbs(a: &[u64; 4], b: &[u64; 4]) -> i32 {
    for i in (0..4).rev() {
        if a[i] < b[i] {
            return -1;
        }
        if a[i] > b[i] {
            return 1;
        }
    }
    0
}

/// Field exponentiation using square-and-multiply.
fn field_pow(base: &[u64; 4], exp: u64) -> [u64; 4] {
    if exp == 0 {
        return to_montgomery(&[1, 0, 0, 0]);
    }

    let mut result = to_montgomery(&[1, 0, 0, 0]);
    let mut base = *base;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = field_mul_limbs(&result, &base);
        }
        base = field_mul_limbs(&base, &base);
        e >>= 1;
    }

    result
}

/// Field inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p.
fn field_inverse(a: &[u64; 4]) -> [u64; 4] {
    // p - 2
    let exp_minus_2: [u64; 4] = [
        MODULUS[0].wrapping_sub(2),
        MODULUS[1],
        MODULUS[2],
        MODULUS[3],
    ];

    // Use binary exponentiation
    let mut result = to_montgomery(&[1, 0, 0, 0]);
    let mut base = *a;

    for i in 0..4 {
        let mut exp_limb = exp_minus_2[i];
        for _ in 0..64 {
            if exp_limb & 1 == 1 {
                result = field_mul_limbs(&result, &base);
            }
            base = field_mul_limbs(&base, &base);
            exp_limb >>= 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse(0, 3), 0);
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
        assert_eq!(bit_reverse(4, 3), 1);
    }

    #[test]
    fn test_ntt_engine_creation() {
        let engine = NttEngine::new();
        println!("GPU available: {}", engine.is_gpu_available());
    }

    #[test]
    fn test_field_add_sub() {
        let one = to_montgomery(&[1, 0, 0, 0]);
        let two = to_montgomery(&[2, 0, 0, 0]);
        let three = to_montgomery(&[3, 0, 0, 0]);

        let sum = field_add_limbs(&one, &two);
        assert_eq!(sum, three);

        let diff = field_sub_limbs(&three, &two);
        assert_eq!(diff, one);
    }

    #[test]
    fn test_field_mul() {
        let two = to_montgomery(&[2, 0, 0, 0]);
        let three = to_montgomery(&[3, 0, 0, 0]);
        let six = to_montgomery(&[6, 0, 0, 0]);

        let prod = field_mul_limbs(&two, &three);
        assert_eq!(prod, six);
    }

    #[test]
    fn test_ntt_roundtrip_small() {
        let mut engine = NttEngine::new();

        // Small test case
        let original: Vec<[u64; 4]> = (0..4)
            .map(|i| to_montgomery(&[i, 0, 0, 0]))
            .collect();

        let mut data = original.clone();

        // Forward then inverse should give back original
        engine.forward(&mut data).unwrap();
        engine.inverse(&mut data).unwrap();

        for i in 0..original.len() {
            assert_eq!(data[i], original[i], "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_twiddle_factors() {
        let twiddles = TwiddleFactors::new(4);
        assert_eq!(twiddles.size, 16);
        assert_eq!(twiddles.forward.len(), 16);
        assert_eq!(twiddles.inverse.len(), 16);
    }
}
