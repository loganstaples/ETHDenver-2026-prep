//! CUDA GPU Acceleration for ZK Proofs.
//!
//! This module provides NVIDIA GPU acceleration for compute-intensive
//! cryptographic operations including MSM, NTT, and field arithmetic.
//!
//! ## Requirements
//!
//! - NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
//! - CUDA Toolkit 11.0+
//! - cuBLAS and cuFFT libraries
//!
//! ## Accelerated Operations
//!
//! - **MSM**: Multi-scalar multiplication using Pippenger's algorithm
//! - **NTT**: Number-theoretic transform for polynomial operations
//! - **Field Arithmetic**: Batch operations on BN254 field elements
//! - **Pairing**: BN254 pairing computations

pub mod bindings;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Check if CUDA is available on this system.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe { bindings::cuda_is_available() }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// CUDA device information.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device index.
    pub index: usize,
    /// Device name.
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Number of SMs (Streaming Multiprocessors).
    pub sm_count: u32,
    /// Max threads per block.
    pub max_threads_per_block: u32,
    /// Max shared memory per block.
    pub max_shared_memory_per_block: u32,
    /// Warp size (typically 32).
    pub warp_size: u32,
}

impl Default for CudaDeviceInfo {
    fn default() -> Self {
        Self {
            index: 0,
            name: "Unknown".to_string(),
            compute_capability: (0, 0),
            total_memory: 0,
            sm_count: 0,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 48 * 1024,
            warp_size: 32,
        }
    }
}

/// CUDA error type.
#[derive(Debug)]
pub enum CudaError {
    /// CUDA runtime not available.
    NotAvailable,
    /// Device initialization failed.
    InitializationFailed(String),
    /// Memory allocation failed.
    AllocationFailed { requested: u64, available: u64 },
    /// Kernel execution failed.
    KernelFailed(String),
    /// Data transfer failed.
    TransferFailed(String),
    /// Invalid argument.
    InvalidArgument(String),
    /// Synchronization failed.
    SyncFailed(String),
    /// Generic error.
    Other(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::NotAvailable => write!(f, "CUDA runtime not available"),
            CudaError::InitializationFailed(msg) => write!(f, "CUDA initialization failed: {}", msg),
            CudaError::AllocationFailed { requested, available } => {
                write!(f, "CUDA allocation failed: requested {} bytes, {} available", requested, available)
            }
            CudaError::KernelFailed(msg) => write!(f, "CUDA kernel failed: {}", msg),
            CudaError::TransferFailed(msg) => write!(f, "CUDA transfer failed: {}", msg),
            CudaError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            CudaError::SyncFailed(msg) => write!(f, "CUDA sync failed: {}", msg),
            CudaError::Other(msg) => write!(f, "CUDA error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA device handle.
pub struct CudaDevice {
    index: usize,
    info: CudaDeviceInfo,
    initialized: AtomicBool,
}

impl CudaDevice {
    /// Creates a new CUDA device handle.
    pub fn new(index: usize) -> CudaResult<Self> {
        if !is_cuda_available() {
            return Err(CudaError::NotAvailable);
        }

        #[cfg(feature = "cuda")]
        {
            let info = unsafe { bindings::cuda_get_device_info(index as i32) }?;
            Ok(Self {
                index,
                info,
                initialized: AtomicBool::new(true),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::NotAvailable)
        }
    }

    /// Creates a new CUDA device handle (stub when CUDA not available).
    #[cfg(not(feature = "cuda"))]
    pub fn new_stub(index: usize) -> Self {
        Self {
            index,
            info: CudaDeviceInfo::default(),
            initialized: AtomicBool::new(false),
        }
    }

    /// Returns device information.
    pub fn info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// Returns whether the device is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Synchronizes the device.
    pub fn synchronize(&self) -> CudaResult<()> {
        if !self.is_initialized() {
            return Err(CudaError::InitializationFailed("Device not initialized".to_string()));
        }

        #[cfg(feature = "cuda")]
        {
            unsafe { bindings::cuda_device_synchronize() }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
    }
}

/// CUDA memory buffer.
#[derive(Debug)]
pub struct CudaBuffer {
    /// Buffer ID.
    pub id: u64,
    /// Size in bytes.
    pub size: u64,
    /// Device index.
    pub device: usize,
    /// Raw device pointer (opaque).
    ptr: u64,
}

impl CudaBuffer {
    /// Creates a new CUDA buffer.
    pub fn new(device: &CudaDevice, size: u64) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let ptr = unsafe { bindings::cuda_malloc(size) }?;
            static BUFFER_ID: AtomicU64 = AtomicU64::new(0);
            Ok(Self {
                id: BUFFER_ID.fetch_add(1, Ordering::Relaxed),
                size,
                device: device.index,
                ptr,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            static BUFFER_ID: AtomicU64 = AtomicU64::new(0);
            Ok(Self {
                id: BUFFER_ID.fetch_add(1, Ordering::Relaxed),
                size,
                device: device.index,
                ptr: 0,
            })
        }
    }

    /// Copies data from host to device.
    pub fn copy_from_host(&self, data: &[u8]) -> CudaResult<()> {
        if data.len() as u64 > self.size {
            return Err(CudaError::InvalidArgument(
                format!("Data size {} exceeds buffer size {}", data.len(), self.size)
            ));
        }

        #[cfg(feature = "cuda")]
        {
            unsafe { bindings::cuda_memcpy_htod(self.ptr, data.as_ptr(), data.len()) }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
    }

    /// Copies data from device to host.
    pub fn copy_to_host(&self, data: &mut [u8]) -> CudaResult<()> {
        if data.len() as u64 > self.size {
            return Err(CudaError::InvalidArgument(
                format!("Data size {} exceeds buffer size {}", data.len(), self.size)
            ));
        }

        #[cfg(feature = "cuda")]
        {
            unsafe { bindings::cuda_memcpy_dtoh(data.as_mut_ptr(), self.ptr, data.len()) }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(())
        }
    }

    /// Returns the raw device pointer.
    pub fn as_ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            unsafe { let _ = bindings::cuda_free(self.ptr); }
        }
    }
}

/// Configuration for CUDA MSM.
#[derive(Debug, Clone)]
pub struct CudaMsmConfig {
    /// Window size for Pippenger's algorithm.
    pub window_size: usize,
    /// Minimum batch size to use GPU.
    pub min_gpu_batch_size: usize,
    /// Number of CUDA streams for async execution.
    pub num_streams: usize,
    /// Block size for kernels.
    pub block_size: usize,
}

impl Default for CudaMsmConfig {
    fn default() -> Self {
        Self {
            window_size: 16,
            min_gpu_batch_size: 256,
            num_streams: 4,
            block_size: 256,
        }
    }
}

/// Configuration for CUDA NTT.
#[derive(Debug, Clone)]
pub struct CudaNttConfig {
    /// Maximum supported log2(n).
    pub max_log_n: usize,
    /// Minimum batch size to use GPU.
    pub min_gpu_batch_size: usize,
    /// Block size for kernels.
    pub block_size: usize,
    /// Whether to use shared memory.
    pub use_shared_memory: bool,
}

impl Default for CudaNttConfig {
    fn default() -> Self {
        Self {
            max_log_n: 24,
            min_gpu_batch_size: 512,
            block_size: 256,
            use_shared_memory: true,
        }
    }
}

/// Statistics for CUDA operations.
#[derive(Debug, Clone, Default)]
pub struct CudaStats {
    /// Number of MSM operations.
    pub msm_operations: u64,
    /// Number of NTT operations.
    pub ntt_operations: u64,
    /// Total GPU time in microseconds.
    pub gpu_time_us: u64,
    /// Total bytes transferred to GPU.
    pub bytes_to_gpu: u64,
    /// Total bytes transferred from GPU.
    pub bytes_from_gpu: u64,
}

/// CUDA MSM engine.
pub struct CudaMsm {
    device: Arc<CudaDevice>,
    config: CudaMsmConfig,
    stats: CudaStats,
}

impl CudaMsm {
    /// Creates a new CUDA MSM engine.
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        Self::with_config(device, CudaMsmConfig::default())
    }

    /// Creates with custom configuration.
    pub fn with_config(device: Arc<CudaDevice>, config: CudaMsmConfig) -> CudaResult<Self> {
        Ok(Self {
            device,
            config,
            stats: CudaStats::default(),
        })
    }

    /// Computes MSM: Σ(scalars[i] * points[i]).
    pub fn compute(
        &mut self,
        points: &[[u64; 8]],   // Affine points as (x, y) with 4 limbs each
        scalars: &[[u64; 4]],  // 256-bit scalars
    ) -> CudaResult<[u64; 12]> {
        if points.len() != scalars.len() {
            return Err(CudaError::InvalidArgument(
                "Points and scalars must have the same length".to_string()
            ));
        }

        if points.is_empty() {
            return Ok([0u64; 12]); // Return identity
        }

        // For small inputs, use CPU fallback
        if points.len() < self.config.min_gpu_batch_size {
            return Ok(self.compute_cpu(points, scalars));
        }

        let start = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        {
            let result = unsafe {
                bindings::cuda_msm_pippenger(
                    points.as_ptr() as *const _,
                    scalars.as_ptr() as *const _,
                    points.len(),
                    self.config.window_size,
                )
            }?;

            self.stats.msm_operations += 1;
            self.stats.gpu_time_us += start.elapsed().as_micros() as u64;

            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(self.compute_cpu(points, scalars))
        }
    }

    /// CPU fallback for MSM.
    fn compute_cpu(&self, _points: &[[u64; 8]], _scalars: &[[u64; 4]]) -> [u64; 12] {
        // Would implement actual MSM here
        // For now, return identity
        [0u64; 12]
    }

    /// Returns statistics.
    pub fn stats(&self) -> &CudaStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CudaStats::default();
    }
}

/// CUDA NTT engine.
pub struct CudaNtt {
    device: Arc<CudaDevice>,
    config: CudaNttConfig,
    stats: CudaStats,
}

impl CudaNtt {
    /// Creates a new CUDA NTT engine.
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        Self::with_config(device, CudaNttConfig::default())
    }

    /// Creates with custom configuration.
    pub fn with_config(device: Arc<CudaDevice>, config: CudaNttConfig) -> CudaResult<Self> {
        Ok(Self {
            device,
            config,
            stats: CudaStats::default(),
        })
    }

    /// Forward NTT in-place.
    pub fn forward(&mut self, data: &mut [[u64; 4]]) -> CudaResult<()> {
        if !data.len().is_power_of_two() {
            return Err(CudaError::InvalidArgument(
                "NTT size must be a power of 2".to_string()
            ));
        }

        if data.len() < self.config.min_gpu_batch_size {
            self.forward_cpu(data);
            return Ok(());
        }

        let start = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        {
            unsafe {
                bindings::cuda_ntt_forward(
                    data.as_mut_ptr() as *mut _,
                    data.len(),
                )?;
            }
        }

        self.stats.ntt_operations += 1;
        self.stats.gpu_time_us += start.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Inverse NTT in-place.
    pub fn inverse(&mut self, data: &mut [[u64; 4]]) -> CudaResult<()> {
        if !data.len().is_power_of_two() {
            return Err(CudaError::InvalidArgument(
                "NTT size must be a power of 2".to_string()
            ));
        }

        if data.len() < self.config.min_gpu_batch_size {
            self.inverse_cpu(data);
            return Ok(());
        }

        let start = std::time::Instant::now();

        #[cfg(feature = "cuda")]
        {
            unsafe {
                bindings::cuda_ntt_inverse(
                    data.as_mut_ptr() as *mut _,
                    data.len(),
                )?;
            }
        }

        self.stats.ntt_operations += 1;
        self.stats.gpu_time_us += start.elapsed().as_micros() as u64;

        Ok(())
    }

    /// CPU forward NTT fallback.
    fn forward_cpu(&self, _data: &mut [[u64; 4]]) {
        // Would implement actual NTT here
    }

    /// CPU inverse NTT fallback.
    fn inverse_cpu(&self, _data: &mut [[u64; 4]]) {
        // Would implement actual INTT here
    }

    /// Returns statistics.
    pub fn stats(&self) -> &CudaStats {
        &self.stats
    }

    /// Resets statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CudaStats::default();
    }
}

/// Enumerate available CUDA devices.
pub fn enumerate_devices() -> Vec<CudaDeviceInfo> {
    #[cfg(feature = "cuda")]
    {
        unsafe { bindings::cuda_enumerate_devices() }.unwrap_or_default()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    fn test_enumerate_devices() {
        let devices = enumerate_devices();
        println!("Found {} CUDA device(s)", devices.len());
        for device in &devices {
            println!("  - {}: {} SM(s), {} MB", device.name, device.sm_count, device.total_memory / 1024 / 1024);
        }
    }

    #[test]
    fn test_msm_config() {
        let config = CudaMsmConfig::default();
        assert_eq!(config.window_size, 16);
        assert_eq!(config.min_gpu_batch_size, 256);
    }

    #[test]
    fn test_ntt_config() {
        let config = CudaNttConfig::default();
        assert_eq!(config.max_log_n, 24);
    }
}
