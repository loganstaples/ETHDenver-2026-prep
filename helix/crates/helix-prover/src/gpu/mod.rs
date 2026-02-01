//! GPU Infrastructure for HELIX.
//!
//! This module provides unified GPU management infrastructure including:
//! - Memory pool management with allocation tracking
//! - Async operation queues with synchronization
//! - Multi-GPU load balancing
//! - Profiling and benchmarking

pub mod pool;
pub mod async_ops;
pub mod multi_gpu;
pub mod hybrid_executor;

pub use pool::{GpuMemoryPool, PooledBuffer, PoolConfig, PoolStats};
pub use async_ops::{AsyncOpQueue, AsyncOp, OpHandle, OpStatus, SyncBarrier};
pub use multi_gpu::{MultiGpuManager, DeviceSelector, WorkDistributor, LoadBalanceStrategy};
pub use hybrid_executor::{HybridExecutor, HybridConfig, HybridStats, WorkItem};

use std::sync::Arc;

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// CPU fallback.
    Cpu,
}

impl std::fmt::Display for GpuBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackendType::Cuda => write!(f, "CUDA"),
            GpuBackendType::Metal => write!(f, "Metal"),
            GpuBackendType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Unified GPU configuration.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred backend type.
    pub preferred_backend: GpuBackendType,
    /// Device index (-1 for auto).
    pub device_index: i32,
    /// Enable multi-GPU.
    pub multi_gpu: bool,
    /// Maximum memory to use per device (0 = unlimited).
    pub max_memory_per_device: u64,
    /// Enable memory pooling.
    pub enable_memory_pool: bool,
    /// Pool size in bytes.
    pub pool_size: u64,
    /// Enable async operations.
    pub enable_async: bool,
    /// Number of async streams/queues.
    pub num_streams: usize,
    /// Enable CPU fallback.
    pub enable_cpu_fallback: bool,
    /// Enable profiling.
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: GpuBackendType::Cuda,
            device_index: -1,
            multi_gpu: false,
            max_memory_per_device: 0,
            enable_memory_pool: true,
            pool_size: 1024 * 1024 * 1024, // 1GB
            enable_async: true,
            num_streams: 4,
            enable_cpu_fallback: true,
            enable_profiling: false,
        }
    }
}

/// Detect available GPU backends.
pub fn detect_backends() -> Vec<(GpuBackendType, usize)> {
    let mut backends = Vec::new();

    // Check CUDA
    #[cfg(feature = "cuda")]
    {
        let cuda_devices = crate::cuda::enumerate_devices().len();
        if cuda_devices > 0 {
            backends.push((GpuBackendType::Cuda, cuda_devices));
        }
    }

    // Check Metal
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        if crate::metal::is_metal_available() {
            backends.push((GpuBackendType::Metal, 1));
        }
    }

    // CPU is always available
    backends.push((GpuBackendType::Cpu, num_cpus::get()));

    backends
}

/// Select best backend based on availability and preferences.
pub fn select_backend(config: &GpuConfig) -> GpuBackendType {
    let backends = detect_backends();

    // Try preferred backend first
    if backends.iter().any(|(t, n)| *t == config.preferred_backend && *n > 0) {
        return config.preferred_backend;
    }

    // Fall back to first available GPU backend
    for (backend, count) in &backends {
        if *backend != GpuBackendType::Cpu && *count > 0 {
            return *backend;
        }
    }

    // CPU fallback
    if config.enable_cpu_fallback {
        GpuBackendType::Cpu
    } else {
        panic!("No GPU backend available and CPU fallback disabled");
    }
}
