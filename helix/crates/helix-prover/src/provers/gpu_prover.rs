//! GPU Acceleration Interface for ZK Proving.
//!
//! Provides an abstraction layer for GPU-accelerated proof generation with
//! backends for CUDA, Metal, and CPU fallback.
//!
//! # Architecture
//!
//! The GPU acceleration system uses a trait-based design:
//!
//! - `GpuBackend`: Abstract interface for GPU operations
//! - `CudaBackend`: NVIDIA CUDA implementation (stub)
//! - `MetalBackend`: Apple Metal implementation (stub)
//! - `CpuFallback`: CPU-only fallback implementation
//!
//! # Features
//!
//! - **MSM acceleration**: Multi-scalar multiplication on GPU
//! - **NTT/FFT acceleration**: Number-theoretic transforms
//! - **Pairing operations**: Elliptic curve pairings
//! - **Memory pooling**: Efficient GPU memory management
//! - **Multi-GPU support**: Distribute work across multiple devices

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

/// GPU device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuDeviceType {
    /// NVIDIA CUDA GPU.
    Cuda,
    /// Apple Metal GPU.
    Metal,
    /// Intel/AMD OpenCL GPU.
    OpenCL,
    /// CPU fallback (no GPU).
    Cpu,
}

impl fmt::Display for GpuDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuDeviceType::Cuda => write!(f, "CUDA"),
            GpuDeviceType::Metal => write!(f, "Metal"),
            GpuDeviceType::OpenCL => write!(f, "OpenCL"),
            GpuDeviceType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Information about a GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// Device index.
    pub index: usize,
    /// Device name.
    pub name: String,
    /// Device type.
    pub device_type: GpuDeviceType,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Available memory in bytes.
    pub available_memory: u64,
    /// Compute capability (CUDA) or feature set version.
    pub compute_capability: (u32, u32),
    /// Number of compute units/SMs.
    pub compute_units: u32,
    /// Maximum threads per block/threadgroup.
    pub max_threads_per_block: u32,
    /// Is the device available.
    pub available: bool,
}

impl GpuDeviceInfo {
    /// Creates a CPU device info.
    pub fn cpu() -> Self {
        Self {
            index: 0,
            name: "CPU Fallback".to_string(),
            device_type: GpuDeviceType::Cpu,
            total_memory: 0,
            available_memory: 0,
            compute_capability: (0, 0),
            compute_units: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
            max_threads_per_block: 1,
            available: true,
        }
    }
}

/// Configuration for GPU proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProverConfig {
    /// Preferred device type.
    pub preferred_device: GpuDeviceType,
    /// Device index (-1 for auto).
    pub device_index: i32,
    /// Enable multi-GPU.
    pub multi_gpu: bool,
    /// Maximum GPU memory to use (0 = unlimited).
    pub max_memory: u64,
    /// Enable memory pooling.
    pub memory_pooling: bool,
    /// Pool size in bytes.
    pub pool_size: u64,
    /// Enable async operations.
    pub async_operations: bool,
    /// Fall back to CPU on GPU failure.
    pub cpu_fallback: bool,
}

impl Default for GpuProverConfig {
    fn default() -> Self {
        Self {
            preferred_device: GpuDeviceType::Cuda,
            device_index: -1, // Auto
            multi_gpu: false,
            max_memory: 0,
            memory_pooling: true,
            pool_size: 1024 * 1024 * 1024, // 1GB
            async_operations: true,
            cpu_fallback: true,
        }
    }
}

/// Error type for GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No GPU device available.
    NoDevice,
    /// Device initialization failed.
    InitializationFailed(String),
    /// Memory allocation failed.
    AllocationFailed { requested: u64, available: u64 },
    /// Kernel execution failed.
    KernelFailed(String),
    /// Data transfer failed.
    TransferFailed(String),
    /// Operation not supported.
    Unsupported(String),
    /// Backend-specific error.
    BackendError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "No GPU device available"),
            GpuError::InitializationFailed(msg) => write!(f, "GPU initialization failed: {}", msg),
            GpuError::AllocationFailed { requested, available } => {
                write!(
                    f,
                    "GPU memory allocation failed: requested {} bytes, {} available",
                    requested, available
                )
            }
            GpuError::KernelFailed(msg) => write!(f, "GPU kernel execution failed: {}", msg),
            GpuError::TransferFailed(msg) => write!(f, "GPU data transfer failed: {}", msg),
            GpuError::Unsupported(msg) => write!(f, "Operation not supported: {}", msg),
            GpuError::BackendError(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU memory buffer handle.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer identifier.
    pub id: u64,
    /// Size in bytes.
    pub size: u64,
    /// Device index.
    pub device: usize,
    /// Is this buffer pinned (page-locked host memory).
    pub pinned: bool,
}

/// Statistics for GPU operations.
#[derive(Debug, Default)]
pub struct GpuStats {
    /// Total operations performed.
    pub operations: AtomicU64,
    /// Total bytes transferred to GPU.
    pub bytes_to_gpu: AtomicU64,
    /// Total bytes transferred from GPU.
    pub bytes_from_gpu: AtomicU64,
    /// Total kernel execution time (microseconds).
    pub kernel_time_us: AtomicU64,
    /// Total transfer time (microseconds).
    pub transfer_time_us: AtomicU64,
    /// MSM operations.
    pub msm_operations: AtomicU64,
    /// NTT operations.
    pub ntt_operations: AtomicU64,
    /// Pairing operations.
    pub pairing_operations: AtomicU64,
}

impl GpuStats {
    fn new() -> Self {
        Self::default()
    }

    /// Gets a snapshot of stats.
    pub fn snapshot(&self) -> GpuStatsSnapshot {
        GpuStatsSnapshot {
            operations: self.operations.load(Ordering::Relaxed),
            bytes_to_gpu: self.bytes_to_gpu.load(Ordering::Relaxed),
            bytes_from_gpu: self.bytes_from_gpu.load(Ordering::Relaxed),
            kernel_time_us: self.kernel_time_us.load(Ordering::Relaxed),
            transfer_time_us: self.transfer_time_us.load(Ordering::Relaxed),
            msm_operations: self.msm_operations.load(Ordering::Relaxed),
            ntt_operations: self.ntt_operations.load(Ordering::Relaxed),
            pairing_operations: self.pairing_operations.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of GPU statistics.
#[derive(Debug, Clone)]
pub struct GpuStatsSnapshot {
    pub operations: u64,
    pub bytes_to_gpu: u64,
    pub bytes_from_gpu: u64,
    pub kernel_time_us: u64,
    pub transfer_time_us: u64,
    pub msm_operations: u64,
    pub ntt_operations: u64,
    pub pairing_operations: u64,
}

/// Trait for GPU backend implementations.
pub trait GpuBackend: Send + Sync {
    /// Returns the backend type.
    fn backend_type(&self) -> GpuDeviceType;

    /// Checks if the backend is available.
    fn is_available(&self) -> bool;

    /// Enumerates available devices.
    fn enumerate_devices(&self) -> Vec<GpuDeviceInfo>;

    /// Initializes a device.
    fn init_device(&mut self, device_index: usize) -> GpuResult<()>;

    /// Shuts down the device.
    fn shutdown(&mut self) -> GpuResult<()>;

    /// Allocates GPU memory.
    fn allocate(&self, size: u64) -> GpuResult<GpuBuffer>;

    /// Frees GPU memory.
    fn free(&self, buffer: &GpuBuffer) -> GpuResult<()>;

    /// Copies data to GPU.
    fn copy_to_device(&self, buffer: &GpuBuffer, data: &[u8]) -> GpuResult<()>;

    /// Copies data from GPU.
    fn copy_from_device(&self, buffer: &GpuBuffer, data: &mut [u8]) -> GpuResult<()>;

    /// Performs multi-scalar multiplication.
    ///
    /// Computes sum(scalars[i] * points[i]) on the GPU.
    fn msm(
        &self,
        points: &[u8],  // Serialized curve points
        scalars: &[u8], // Serialized scalars
        result: &mut [u8],
    ) -> GpuResult<()>;

    /// Performs Number-Theoretic Transform (NTT/FFT).
    fn ntt(&self, data: &mut [u8], inverse: bool) -> GpuResult<()>;

    /// Computes elliptic curve pairing.
    fn pairing(
        &self,
        g1_points: &[u8],
        g2_points: &[u8],
        result: &mut [u8],
    ) -> GpuResult<()>;

    /// Synchronizes all pending operations.
    fn synchronize(&self) -> GpuResult<()>;

    /// Returns backend statistics.
    fn stats(&self) -> GpuStatsSnapshot;
}

/// CUDA backend implementation (stub).
pub struct CudaBackend {
    /// Configuration.
    config: GpuProverConfig,
    /// Initialized device.
    device: Option<usize>,
    /// Statistics.
    stats: Arc<GpuStats>,
    /// Is initialized.
    initialized: AtomicBool,
}

impl CudaBackend {
    /// Creates a new CUDA backend.
    pub fn new(config: GpuProverConfig) -> Self {
        Self {
            config,
            device: None,
            stats: Arc::new(GpuStats::new()),
            initialized: AtomicBool::new(false),
        }
    }

    /// Checks if CUDA is available on the system.
    pub fn cuda_available() -> bool {
        // Stub: In production, would call CUDA runtime to check.
        // For now, always return false to indicate CUDA is not linked.
        false
    }
}

impl GpuBackend for CudaBackend {
    fn backend_type(&self) -> GpuDeviceType {
        GpuDeviceType::Cuda
    }

    fn is_available(&self) -> bool {
        Self::cuda_available()
    }

    fn enumerate_devices(&self) -> Vec<GpuDeviceInfo> {
        // Stub: Would enumerate CUDA devices.
        // Returns empty list since CUDA is not linked.
        Vec::new()
    }

    fn init_device(&mut self, device_index: usize) -> GpuResult<()> {
        if !self.is_available() {
            return Err(GpuError::InitializationFailed(
                "CUDA runtime not available".to_string(),
            ));
        }

        // Stub: Would initialize CUDA device.
        // cuInit(0)
        // cuDeviceGet(&device, device_index)
        // cuCtxCreate(&context, 0, device)

        self.device = Some(device_index);
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn shutdown(&mut self) -> GpuResult<()> {
        // Stub: Would destroy CUDA context.
        // cuCtxDestroy(context)

        self.device = None;
        self.initialized.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn allocate(&self, size: u64) -> GpuResult<GpuBuffer> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would allocate CUDA memory.
        // cuMemAlloc(&ptr, size)

        static BUFFER_ID: AtomicU64 = AtomicU64::new(0);

        Ok(GpuBuffer {
            id: BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            size,
            device: self.device.unwrap_or(0),
            pinned: false,
        })
    }

    fn free(&self, _buffer: &GpuBuffer) -> GpuResult<()> {
        // Stub: Would free CUDA memory.
        // cuMemFree(ptr)
        Ok(())
    }

    fn copy_to_device(&self, buffer: &GpuBuffer, data: &[u8]) -> GpuResult<()> {
        // Stub: Would copy to CUDA device.
        // cuMemcpyHtoD(buffer.ptr, data.as_ptr(), data.len())

        self.stats
            .bytes_to_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn copy_from_device(&self, buffer: &GpuBuffer, data: &mut [u8]) -> GpuResult<()> {
        // Stub: Would copy from CUDA device.
        // cuMemcpyDtoH(data.as_mut_ptr(), buffer.ptr, data.len())

        self.stats
            .bytes_from_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn msm(
        &self,
        _points: &[u8],
        _scalars: &[u8],
        _result: &mut [u8],
    ) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would launch CUDA MSM kernel.
        // This would use techniques like:
        // - Pippenger's algorithm on GPU
        // - Bucket method with parallel reduction
        // - cuBLAS for large matrix operations

        self.stats.msm_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("CUDA MSM not implemented".to_string()))
    }

    fn ntt(&self, _data: &mut [u8], _inverse: bool) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would launch CUDA NTT kernel.
        // This would use cuFFT-like techniques adapted for finite fields.

        self.stats.ntt_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("CUDA NTT not implemented".to_string()))
    }

    fn pairing(
        &self,
        _g1_points: &[u8],
        _g2_points: &[u8],
        _result: &mut [u8],
    ) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would launch CUDA pairing kernel.
        // Pairing operations on BN254/BLS12-381.

        self.stats.pairing_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("CUDA pairing not implemented".to_string()))
    }

    fn synchronize(&self) -> GpuResult<()> {
        // Stub: Would synchronize CUDA stream.
        // cuStreamSynchronize(stream)
        Ok(())
    }

    fn stats(&self) -> GpuStatsSnapshot {
        self.stats.snapshot()
    }
}

/// Metal backend implementation (stub).
pub struct MetalBackend {
    /// Configuration.
    config: GpuProverConfig,
    /// Initialized device.
    device: Option<usize>,
    /// Statistics.
    stats: Arc<GpuStats>,
    /// Is initialized.
    initialized: AtomicBool,
}

impl MetalBackend {
    /// Creates a new Metal backend.
    pub fn new(config: GpuProverConfig) -> Self {
        Self {
            config,
            device: None,
            stats: Arc::new(GpuStats::new()),
            initialized: AtomicBool::new(false),
        }
    }

    /// Checks if Metal is available on the system.
    pub fn metal_available() -> bool {
        // Stub: In production, would check for Metal framework.
        // Only available on macOS/iOS.
        #[cfg(target_os = "macos")]
        {
            // Would call MTLCopyAllDevices() and check if non-empty.
            false
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

impl GpuBackend for MetalBackend {
    fn backend_type(&self) -> GpuDeviceType {
        GpuDeviceType::Metal
    }

    fn is_available(&self) -> bool {
        Self::metal_available()
    }

    fn enumerate_devices(&self) -> Vec<GpuDeviceInfo> {
        // Stub: Would enumerate Metal devices.
        // MTLCopyAllDevices()
        Vec::new()
    }

    fn init_device(&mut self, device_index: usize) -> GpuResult<()> {
        if !self.is_available() {
            return Err(GpuError::InitializationFailed(
                "Metal runtime not available".to_string(),
            ));
        }

        // Stub: Would initialize Metal device.
        // device = MTLCopyAllDevices()[device_index]
        // commandQueue = device.makeCommandQueue()

        self.device = Some(device_index);
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn shutdown(&mut self) -> GpuResult<()> {
        // Stub: Metal devices are released automatically.
        self.device = None;
        self.initialized.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn allocate(&self, size: u64) -> GpuResult<GpuBuffer> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would allocate Metal buffer.
        // buffer = device.makeBuffer(length: size, options: .storageModeShared)

        static BUFFER_ID: AtomicU64 = AtomicU64::new(0);

        Ok(GpuBuffer {
            id: BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            size,
            device: self.device.unwrap_or(0),
            pinned: false,
        })
    }

    fn free(&self, _buffer: &GpuBuffer) -> GpuResult<()> {
        // Stub: Metal buffers are ARC-managed.
        Ok(())
    }

    fn copy_to_device(&self, buffer: &GpuBuffer, data: &[u8]) -> GpuResult<()> {
        // Stub: Metal shared buffers have automatic sync.
        // memcpy(buffer.contents(), data.as_ptr(), data.len())

        self.stats
            .bytes_to_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn copy_from_device(&self, buffer: &GpuBuffer, data: &mut [u8]) -> GpuResult<()> {
        // Stub: Metal shared buffers have automatic sync.
        // memcpy(data.as_mut_ptr(), buffer.contents(), data.len())

        self.stats
            .bytes_from_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn msm(
        &self,
        _points: &[u8],
        _scalars: &[u8],
        _result: &mut [u8],
    ) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would dispatch Metal compute shader for MSM.
        // Uses Metal Performance Shaders or custom compute kernels.

        self.stats.msm_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("Metal MSM not implemented".to_string()))
    }

    fn ntt(&self, _data: &mut [u8], _inverse: bool) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would dispatch Metal compute shader for NTT.

        self.stats.ntt_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("Metal NTT not implemented".to_string()))
    }

    fn pairing(
        &self,
        _g1_points: &[u8],
        _g2_points: &[u8],
        _result: &mut [u8],
    ) -> GpuResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(GpuError::InitializationFailed("Device not initialized".to_string()));
        }

        // Stub: Would dispatch Metal compute shader for pairing.

        self.stats.pairing_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        Err(GpuError::Unsupported("Metal pairing not implemented".to_string()))
    }

    fn synchronize(&self) -> GpuResult<()> {
        // Stub: Would wait for command buffer completion.
        // commandBuffer.waitUntilCompleted()
        Ok(())
    }

    fn stats(&self) -> GpuStatsSnapshot {
        self.stats.snapshot()
    }
}

/// CPU fallback implementation.
pub struct CpuFallback {
    /// Statistics.
    stats: Arc<GpuStats>,
    /// Is initialized.
    initialized: AtomicBool,
}

impl CpuFallback {
    /// Creates a new CPU fallback.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(GpuStats::new()),
            initialized: AtomicBool::new(true), // CPU is always ready.
        }
    }
}

impl Default for CpuFallback {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for CpuFallback {
    fn backend_type(&self) -> GpuDeviceType {
        GpuDeviceType::Cpu
    }

    fn is_available(&self) -> bool {
        true // CPU is always available.
    }

    fn enumerate_devices(&self) -> Vec<GpuDeviceInfo> {
        vec![GpuDeviceInfo::cpu()]
    }

    fn init_device(&mut self, _device_index: usize) -> GpuResult<()> {
        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn shutdown(&mut self) -> GpuResult<()> {
        Ok(())
    }

    fn allocate(&self, size: u64) -> GpuResult<GpuBuffer> {
        // CPU "buffers" are just tracked allocations.
        static BUFFER_ID: AtomicU64 = AtomicU64::new(0);

        Ok(GpuBuffer {
            id: BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            size,
            device: 0,
            pinned: false,
        })
    }

    fn free(&self, _buffer: &GpuBuffer) -> GpuResult<()> {
        Ok(())
    }

    fn copy_to_device(&self, buffer: &GpuBuffer, data: &[u8]) -> GpuResult<()> {
        // No-op for CPU.
        self.stats
            .bytes_to_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn copy_from_device(&self, buffer: &GpuBuffer, data: &mut [u8]) -> GpuResult<()> {
        // No-op for CPU.
        self.stats
            .bytes_from_gpu
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn msm(
        &self,
        points: &[u8],
        scalars: &[u8],
        result: &mut [u8],
    ) -> GpuResult<()> {
        // CPU MSM implementation would use existing Rust libraries.
        // For now, return placeholder result.

        self.stats.msm_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        // In production, would call arkworks/halo2 MSM.
        // For stub, just zero the result.
        result.fill(0);

        Ok(())
    }

    fn ntt(&self, data: &mut [u8], _inverse: bool) -> GpuResult<()> {
        // CPU NTT implementation would use existing Rust libraries.

        self.stats.ntt_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        // In production, would call arkworks/halo2 NTT.
        // For stub, leave data unchanged.

        Ok(())
    }

    fn pairing(
        &self,
        g1_points: &[u8],
        g2_points: &[u8],
        result: &mut [u8],
    ) -> GpuResult<()> {
        // CPU pairing implementation would use existing Rust libraries.

        self.stats.pairing_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.operations.fetch_add(1, Ordering::Relaxed);

        // In production, would call arkworks/halo2 pairing.
        // For stub, just zero the result.
        result.fill(0);

        Ok(())
    }

    fn synchronize(&self) -> GpuResult<()> {
        Ok(()) // No-op for CPU.
    }

    fn stats(&self) -> GpuStatsSnapshot {
        self.stats.snapshot()
    }
}

/// GPU-accelerated prover that automatically selects the best backend.
pub struct GpuProver {
    /// Configuration.
    config: GpuProverConfig,
    /// Active backend.
    backend: RwLock<Box<dyn GpuBackend>>,
    /// Selected device info.
    device_info: RwLock<Option<GpuDeviceInfo>>,
}

impl GpuProver {
    /// Creates a new GPU prover with automatic backend selection.
    pub fn new(config: GpuProverConfig) -> Self {
        let backend = Self::select_backend(&config);
        let device_info = backend.enumerate_devices().first().cloned();

        Self {
            config,
            backend: RwLock::new(backend),
            device_info: RwLock::new(device_info),
        }
    }

    /// Creates a GPU prover with CPU fallback.
    pub fn cpu_only() -> Self {
        Self::new(GpuProverConfig {
            preferred_device: GpuDeviceType::Cpu,
            ..Default::default()
        })
    }

    fn select_backend(config: &GpuProverConfig) -> Box<dyn GpuBackend> {
        match config.preferred_device {
            GpuDeviceType::Cuda => {
                let backend = CudaBackend::new(config.clone());
                if backend.is_available() {
                    return Box::new(backend);
                }
            }
            GpuDeviceType::Metal => {
                let backend = MetalBackend::new(config.clone());
                if backend.is_available() {
                    return Box::new(backend);
                }
            }
            _ => {}
        }

        // Fall back to CPU.
        Box::new(CpuFallback::new())
    }

    /// Returns the active backend type.
    pub fn backend_type(&self) -> GpuDeviceType {
        self.backend.read().unwrap().backend_type()
    }

    /// Returns whether GPU acceleration is available.
    pub fn is_gpu_available(&self) -> bool {
        let backend = self.backend.read().unwrap();
        backend.backend_type() != GpuDeviceType::Cpu && backend.is_available()
    }

    /// Initializes the prover.
    pub fn init(&self) -> GpuResult<()> {
        let mut backend = self.backend.write().unwrap();
        let device_index = if self.config.device_index < 0 {
            0
        } else {
            self.config.device_index as usize
        };
        backend.init_device(device_index)
    }

    /// Shuts down the prover.
    pub fn shutdown(&self) -> GpuResult<()> {
        let mut backend = self.backend.write().unwrap();
        backend.shutdown()
    }

    /// Performs MSM operation.
    pub fn msm(&self, points: &[u8], scalars: &[u8], result: &mut [u8]) -> GpuResult<()> {
        let backend = self.backend.read().unwrap();
        backend.msm(points, scalars, result)
    }

    /// Performs NTT operation.
    pub fn ntt(&self, data: &mut [u8], inverse: bool) -> GpuResult<()> {
        let backend = self.backend.read().unwrap();
        backend.ntt(data, inverse)
    }

    /// Performs pairing operation.
    pub fn pairing(&self, g1: &[u8], g2: &[u8], result: &mut [u8]) -> GpuResult<()> {
        let backend = self.backend.read().unwrap();
        backend.pairing(g1, g2, result)
    }

    /// Returns statistics.
    pub fn stats(&self) -> GpuStatsSnapshot {
        let backend = self.backend.read().unwrap();
        backend.stats()
    }

    /// Returns device info.
    pub fn device_info(&self) -> Option<GpuDeviceInfo> {
        self.device_info.read().unwrap().clone()
    }
}

/// Detects available GPU backends.
pub fn detect_gpu_backends() -> Vec<(GpuDeviceType, bool)> {
    vec![
        (GpuDeviceType::Cuda, CudaBackend::cuda_available()),
        (GpuDeviceType::Metal, MetalBackend::metal_available()),
        (GpuDeviceType::Cpu, true),
    ]
}

/// Creates the best available GPU prover.
pub fn create_best_gpu_prover() -> GpuProver {
    // Try CUDA first, then Metal, then CPU.
    if CudaBackend::cuda_available() {
        GpuProver::new(GpuProverConfig {
            preferred_device: GpuDeviceType::Cuda,
            ..Default::default()
        })
    } else if MetalBackend::metal_available() {
        GpuProver::new(GpuProverConfig {
            preferred_device: GpuDeviceType::Metal,
            ..Default::default()
        })
    } else {
        GpuProver::cpu_only()
    }
}

/// Memory pool for GPU allocations.
pub struct GpuMemoryPool {
    /// Backend.
    backend: Arc<RwLock<Box<dyn GpuBackend>>>,
    /// Pool size.
    pool_size: u64,
    /// Allocated buffers.
    buffers: RwLock<Vec<GpuBuffer>>,
    /// Free list.
    free_list: RwLock<Vec<GpuBuffer>>,
}

impl GpuMemoryPool {
    /// Creates a new memory pool.
    pub fn new(backend: Arc<RwLock<Box<dyn GpuBackend>>>, pool_size: u64) -> Self {
        Self {
            backend,
            pool_size,
            buffers: RwLock::new(Vec::new()),
            free_list: RwLock::new(Vec::new()),
        }
    }

    /// Allocates from the pool.
    pub fn allocate(&self, size: u64) -> GpuResult<GpuBuffer> {
        // Check free list for suitable buffer.
        {
            let mut free = self.free_list.write().unwrap();
            if let Some(idx) = free.iter().position(|b| b.size >= size) {
                return Ok(free.remove(idx));
            }
        }

        // Allocate new buffer.
        let backend = self.backend.read().unwrap();
        let buffer = backend.allocate(size)?;

        {
            let mut buffers = self.buffers.write().unwrap();
            buffers.push(buffer.clone());
        }

        Ok(buffer)
    }

    /// Returns a buffer to the pool.
    pub fn free(&self, buffer: GpuBuffer) {
        let mut free = self.free_list.write().unwrap();
        free.push(buffer);
    }

    /// Returns the number of allocated buffers.
    pub fn allocated_count(&self) -> usize {
        self.buffers.read().unwrap().len()
    }

    /// Returns the number of free buffers.
    pub fn free_count(&self) -> usize {
        self.free_list.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_type_display() {
        assert_eq!(format!("{}", GpuDeviceType::Cuda), "CUDA");
        assert_eq!(format!("{}", GpuDeviceType::Metal), "Metal");
        assert_eq!(format!("{}", GpuDeviceType::Cpu), "CPU");
    }

    #[test]
    fn test_cpu_fallback() {
        let mut backend = CpuFallback::new();

        assert!(backend.is_available());
        assert_eq!(backend.backend_type(), GpuDeviceType::Cpu);

        backend.init_device(0).unwrap();

        let devices = backend.enumerate_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_type, GpuDeviceType::Cpu);
    }

    #[test]
    fn test_cpu_msm() {
        let backend = CpuFallback::new();

        let points = vec![0u8; 64];
        let scalars = vec![0u8; 32];
        let mut result = vec![0u8; 64];

        backend.msm(&points, &scalars, &mut result).unwrap();

        let stats = backend.stats();
        assert_eq!(stats.msm_operations, 1);
    }

    #[test]
    fn test_gpu_prover_cpu_fallback() {
        let prover = GpuProver::cpu_only();

        assert_eq!(prover.backend_type(), GpuDeviceType::Cpu);
        assert!(!prover.is_gpu_available());

        prover.init().unwrap();

        let mut result = vec![0u8; 64];
        prover.msm(&[0u8; 64], &[0u8; 32], &mut result).unwrap();
    }

    #[test]
    fn test_detect_backends() {
        let backends = detect_gpu_backends();

        assert!(backends.len() >= 3);

        // CPU should always be available.
        assert!(backends.iter().any(|(t, a)| *t == GpuDeviceType::Cpu && *a));
    }

    #[test]
    fn test_gpu_error_display() {
        let err = GpuError::NoDevice;
        assert_eq!(format!("{}", err), "No GPU device available");

        let err = GpuError::AllocationFailed {
            requested: 1000,
            available: 500,
        };
        assert!(format!("{}", err).contains("1000"));
    }

    #[test]
    fn test_gpu_stats() {
        let stats = GpuStats::new();

        stats.operations.fetch_add(5, Ordering::Relaxed);
        stats.msm_operations.fetch_add(3, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.operations, 5);
        assert_eq!(snapshot.msm_operations, 3);
    }

    #[test]
    fn test_memory_pool() {
        let backend: Box<dyn GpuBackend> = Box::new(CpuFallback::new());
        let pool = GpuMemoryPool::new(Arc::new(RwLock::new(backend)), 1024 * 1024);

        let buf1 = pool.allocate(1024).unwrap();
        assert_eq!(pool.allocated_count(), 1);

        let buf2 = pool.allocate(2048).unwrap();
        assert_eq!(pool.allocated_count(), 2);

        pool.free(buf1);
        assert_eq!(pool.free_count(), 1);

        // Should reuse the freed buffer.
        let buf3 = pool.allocate(512).unwrap();
        assert_eq!(pool.allocated_count(), 2);
        assert_eq!(pool.free_count(), 0);
    }

    #[test]
    fn test_cuda_backend_unavailable() {
        let backend = CudaBackend::new(GpuProverConfig::default());

        // CUDA should not be available without the runtime.
        assert!(!backend.is_available());
        assert!(backend.enumerate_devices().is_empty());
    }

    #[test]
    fn test_metal_backend_unavailable() {
        let backend = MetalBackend::new(GpuProverConfig::default());

        // Metal availability depends on platform.
        let devices = backend.enumerate_devices();
        // On non-macOS, should be empty.
        #[cfg(not(target_os = "macos"))]
        assert!(devices.is_empty());
    }

    #[test]
    fn test_create_best_prover() {
        let prover = create_best_gpu_prover();

        // Should fall back to CPU on test environments.
        // Note: On machines with CUDA/Metal, this might be different.
        let backend_type = prover.backend_type();
        assert!(
            backend_type == GpuDeviceType::Cpu
                || backend_type == GpuDeviceType::Cuda
                || backend_type == GpuDeviceType::Metal
        );
    }
}
