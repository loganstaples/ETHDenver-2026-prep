//! CUDA FFI Bindings.
//!
//! This module provides Rust bindings to the CUDA runtime and custom kernels.
//! The actual CUDA kernels are in msm.cu and ntt.cu.

use super::{CudaDeviceInfo, CudaError, CudaResult};

// ============================================================================
// FFI Declarations
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    // ========================================================================
    // CUDA Runtime API
    // ========================================================================

    /// Initialize CUDA runtime.
    fn helix_cuda_init() -> i32;

    /// Check if CUDA is available.
    fn helix_cuda_is_available() -> i32;

    /// Get device count.
    fn helix_cuda_get_device_count() -> i32;

    /// Get device properties.
    fn helix_cuda_get_device_info(
        device: i32,
        name: *mut u8,
        name_len: i32,
        compute_major: *mut i32,
        compute_minor: *mut i32,
        total_memory: *mut u64,
        sm_count: *mut i32,
        max_threads: *mut i32,
        max_shared_mem: *mut i32,
        warp_size: *mut i32,
    ) -> i32;

    /// Set current device.
    fn helix_cuda_set_device(device: i32) -> i32;

    /// Synchronize device.
    fn helix_cuda_device_synchronize() -> i32;

    // ========================================================================
    // Memory Management
    // ========================================================================

    /// Allocate device memory.
    fn helix_cuda_malloc(size: u64, ptr: *mut u64) -> i32;

    /// Free device memory.
    fn helix_cuda_free(ptr: u64) -> i32;

    /// Copy host to device.
    fn helix_cuda_memcpy_htod(dst: u64, src: *const u8, size: usize) -> i32;

    /// Copy device to host.
    fn helix_cuda_memcpy_dtoh(dst: *mut u8, src: u64, size: usize) -> i32;

    /// Copy device to device.
    fn helix_cuda_memcpy_dtod(dst: u64, src: u64, size: usize) -> i32;

    /// Set device memory.
    fn helix_cuda_memset(ptr: u64, value: i32, size: usize) -> i32;

    // ========================================================================
    // MSM Kernels
    // ========================================================================

    /// MSM using Pippenger's algorithm.
    fn helix_cuda_msm_pippenger(
        points: *const u64,      // Array of affine points
        scalars: *const u64,     // Array of scalars
        count: usize,            // Number of points/scalars
        window_size: usize,      // Window size for Pippenger
        result: *mut u64,        // Result point (projective)
    ) -> i32;

    /// Batch MSM for multiple independent MSMs.
    fn helix_cuda_msm_batch(
        points: *const u64,
        scalars: *const u64,
        counts: *const usize,    // Array of counts per MSM
        num_msms: usize,         // Number of MSMs
        window_size: usize,
        results: *mut u64,       // Array of results
    ) -> i32;

    // ========================================================================
    // NTT Kernels
    // ========================================================================

    /// Forward NTT.
    fn helix_cuda_ntt_forward(
        data: *mut u64,          // In-place data
        count: usize,            // Must be power of 2
    ) -> i32;

    /// Inverse NTT.
    fn helix_cuda_ntt_inverse(
        data: *mut u64,
        count: usize,
    ) -> i32;

    /// Batch NTT (multiple independent NTTs).
    fn helix_cuda_ntt_batch(
        data: *mut u64,
        count: usize,            // Elements per NTT
        batch_size: usize,       // Number of NTTs
        inverse: i32,            // 0 = forward, 1 = inverse
    ) -> i32;

    // ========================================================================
    // Field Arithmetic Kernels
    // ========================================================================

    /// Batch field addition.
    fn helix_cuda_field_add(
        a: *const u64,
        b: *const u64,
        c: *mut u64,
        count: usize,
    ) -> i32;

    /// Batch field subtraction.
    fn helix_cuda_field_sub(
        a: *const u64,
        b: *const u64,
        c: *mut u64,
        count: usize,
    ) -> i32;

    /// Batch field multiplication.
    fn helix_cuda_field_mul(
        a: *const u64,
        b: *const u64,
        c: *mut u64,
        count: usize,
    ) -> i32;

    /// Batch field inversion using Montgomery's trick.
    fn helix_cuda_field_batch_inv(
        a: *const u64,
        inv: *mut u64,
        count: usize,
    ) -> i32;

    // ========================================================================
    // Polynomial Operations
    // ========================================================================

    /// Polynomial evaluation at multiple points.
    fn helix_cuda_poly_eval_multi(
        coeffs: *const u64,      // Polynomial coefficients
        degree: usize,
        points: *const u64,      // Evaluation points
        results: *mut u64,       // Results
        num_points: usize,
    ) -> i32;

    /// Polynomial multiplication via NTT.
    fn helix_cuda_poly_mul(
        a: *const u64,
        a_len: usize,
        b: *const u64,
        b_len: usize,
        result: *mut u64,
        result_len: usize,
    ) -> i32;
}

// ============================================================================
// Safe Wrappers
// ============================================================================

/// Check if CUDA is available.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_is_available() -> bool {
    helix_cuda_is_available() != 0
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_is_available() -> bool {
    false
}

/// Get device info.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_get_device_info(device: i32) -> CudaResult<CudaDeviceInfo> {
    let mut name = [0u8; 256];
    let mut compute_major = 0i32;
    let mut compute_minor = 0i32;
    let mut total_memory = 0u64;
    let mut sm_count = 0i32;
    let mut max_threads = 0i32;
    let mut max_shared_mem = 0i32;
    let mut warp_size = 0i32;

    let result = helix_cuda_get_device_info(
        device,
        name.as_mut_ptr(),
        name.len() as i32,
        &mut compute_major,
        &mut compute_minor,
        &mut total_memory,
        &mut sm_count,
        &mut max_threads,
        &mut max_shared_mem,
        &mut warp_size,
    );

    if result != 0 {
        return Err(CudaError::InitializationFailed(
            format!("Failed to get device info: error {}", result)
        ));
    }

    let name_str = std::ffi::CStr::from_ptr(name.as_ptr() as *const _)
        .to_string_lossy()
        .into_owned();

    Ok(CudaDeviceInfo {
        index: device as usize,
        name: name_str,
        compute_capability: (compute_major as u32, compute_minor as u32),
        total_memory,
        sm_count: sm_count as u32,
        max_threads_per_block: max_threads as u32,
        max_shared_memory_per_block: max_shared_mem as u32,
        warp_size: warp_size as u32,
    })
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_get_device_info(_device: i32) -> CudaResult<CudaDeviceInfo> {
    Err(CudaError::NotAvailable)
}

/// Enumerate all CUDA devices.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_enumerate_devices() -> CudaResult<Vec<CudaDeviceInfo>> {
    let count = helix_cuda_get_device_count();
    if count <= 0 {
        return Ok(Vec::new());
    }

    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        if let Ok(info) = cuda_get_device_info(i) {
            devices.push(info);
        }
    }
    Ok(devices)
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_enumerate_devices() -> CudaResult<Vec<CudaDeviceInfo>> {
    Ok(Vec::new())
}

/// Synchronize device.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_device_synchronize() -> CudaResult<()> {
    let result = helix_cuda_device_synchronize();
    if result != 0 {
        return Err(CudaError::SyncFailed(format!("CUDA sync error: {}", result)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_device_synchronize() -> CudaResult<()> {
    Ok(())
}

/// Allocate device memory.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_malloc(size: u64) -> CudaResult<u64> {
    let mut ptr = 0u64;
    let result = helix_cuda_malloc(size, &mut ptr);
    if result != 0 {
        return Err(CudaError::AllocationFailed {
            requested: size,
            available: 0,
        });
    }
    Ok(ptr)
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_malloc(_size: u64) -> CudaResult<u64> {
    Err(CudaError::NotAvailable)
}

/// Free device memory.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_free(ptr: u64) -> CudaResult<()> {
    let result = helix_cuda_free(ptr);
    if result != 0 {
        return Err(CudaError::Other(format!("Failed to free memory: {}", result)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_free(_ptr: u64) -> CudaResult<()> {
    Ok(())
}

/// Copy host to device.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_htod(dst: u64, src: *const u8, size: usize) -> CudaResult<()> {
    let result = helix_cuda_memcpy_htod(dst, src, size);
    if result != 0 {
        return Err(CudaError::TransferFailed(format!("H->D copy failed: {}", result)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_memcpy_htod(_dst: u64, _src: *const u8, _size: usize) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// Copy device to host.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_memcpy_dtoh(dst: *mut u8, src: u64, size: usize) -> CudaResult<()> {
    let result = helix_cuda_memcpy_dtoh(dst, src, size);
    if result != 0 {
        return Err(CudaError::TransferFailed(format!("D->H copy failed: {}", result)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_memcpy_dtoh(_dst: *mut u8, _src: u64, _size: usize) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// MSM using Pippenger's algorithm.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_msm_pippenger(
    points: *const u64,
    scalars: *const u64,
    count: usize,
    window_size: usize,
) -> CudaResult<[u64; 12]> {
    let mut result = [0u64; 12];
    let status = helix_cuda_msm_pippenger(
        points,
        scalars,
        count,
        window_size,
        result.as_mut_ptr(),
    );
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("MSM failed: {}", status)));
    }
    Ok(result)
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_msm_pippenger(
    _points: *const u64,
    _scalars: *const u64,
    _count: usize,
    _window_size: usize,
) -> CudaResult<[u64; 12]> {
    Err(CudaError::NotAvailable)
}

/// Forward NTT.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_ntt_forward(data: *mut u64, count: usize) -> CudaResult<()> {
    let status = helix_cuda_ntt_forward(data, count);
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("NTT forward failed: {}", status)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_ntt_forward(_data: *mut u64, _count: usize) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// Inverse NTT.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_ntt_inverse(data: *mut u64, count: usize) -> CudaResult<()> {
    let status = helix_cuda_ntt_inverse(data, count);
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("NTT inverse failed: {}", status)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_ntt_inverse(_data: *mut u64, _count: usize) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// Batch field addition.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_field_add(
    a: *const u64,
    b: *const u64,
    c: *mut u64,
    count: usize,
) -> CudaResult<()> {
    let status = helix_cuda_field_add(a, b, c, count);
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("Field add failed: {}", status)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_field_add(
    _a: *const u64,
    _b: *const u64,
    _c: *mut u64,
    _count: usize,
) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// Batch field multiplication.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_field_mul(
    a: *const u64,
    b: *const u64,
    c: *mut u64,
    count: usize,
) -> CudaResult<()> {
    let status = helix_cuda_field_mul(a, b, c, count);
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("Field mul failed: {}", status)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_field_mul(
    _a: *const u64,
    _b: *const u64,
    _c: *mut u64,
    _count: usize,
) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}

/// Batch field inversion.
#[cfg(feature = "cuda")]
pub unsafe fn cuda_field_batch_inv(
    a: *const u64,
    inv: *mut u64,
    count: usize,
) -> CudaResult<()> {
    let status = helix_cuda_field_batch_inv(a, inv, count);
    if status != 0 {
        return Err(CudaError::KernelFailed(format!("Batch inv failed: {}", status)));
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_field_batch_inv(
    _a: *const u64,
    _inv: *mut u64,
    _count: usize,
) -> CudaResult<()> {
    Err(CudaError::NotAvailable)
}
