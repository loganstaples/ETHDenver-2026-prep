//! Metal GPU Acceleration for GKR Proofs.
//!
//! This module provides GPU acceleration for the compute-intensive parts of
//! the GKR protocol using Apple Metal. It is only available on macOS.
//!
//! ## Accelerated Operations
//!
//! - **Field arithmetic**: Batch field operations (add, mul, inv)
//! - **Polynomial evaluation**: Multilinear extension evaluation
//! - **Sumcheck**: Parallel partial sum computation
//! - **Matrix operations**: Optimized for neural network circuits
//! - **MSM**: Multi-scalar multiplication using Pippenger's algorithm
//! - **NTT**: Number-theoretic transform for polynomial operations
//!
//! ## Usage
//!
//! ```ignore
//! use helix_prover::metal::{MetalDevice, MsmEngine, NttEngine};
//!
//! // MSM computation
//! let mut msm = MsmEngine::new();
//! let result = msm.compute(&points, &scalars)?;
//!
//! // NTT computation
//! let mut ntt = NttEngine::new();
//! ntt.forward(&mut coefficients)?;
//! ntt.inverse(&mut coefficients)?;
//! ```

pub mod device;
pub mod field_ops;
pub mod msm;
pub mod ntt;

// Conditional compilation for Metal
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod shaders;

pub use device::{MetalDevice, MetalDeviceInfo, MetalError, MetalResult, BufferPool};
pub use field_ops::{
    MetalFieldOps, MetalPolynomialOps, MetalSumcheckAccelerator,
    BatchFieldOperation, FieldOpType,
};
pub use msm::{
    MetalMsm, MsmEngine, MsmConfig, MsmStats,
    AffinePoint, ProjectivePoint, Scalar,
};
pub use ntt::{
    MetalNtt, NttEngine, NttConfig, NttStats, TwiddleFactors,
};

use super::gkr::FieldElement;

/// Check if Metal acceleration is available.
pub fn is_metal_available() -> bool {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        MetalDevice::is_available()
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        false
    }
}

/// Get information about the Metal device.
pub fn get_device_info() -> Option<MetalDeviceInfo> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        MetalDevice::new().ok().map(|d| d.info())
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        None
    }
}

/// Configuration for Metal acceleration.
#[derive(Debug, Clone)]
pub struct MetalConfig {
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Number of threadgroups for compute operations.
    pub threadgroup_size: usize,
    /// Whether to use shared memory optimization.
    pub use_shared_memory: bool,
    /// Whether to enable profiling.
    pub enable_profiling: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 256 * 1024 * 1024, // 256 MB
            threadgroup_size: 256,
            use_shared_memory: true,
            enable_profiling: false,
        }
    }
}

/// Statistics from Metal operations.
#[derive(Debug, Clone, Default)]
pub struct MetalStats {
    /// Total GPU time in microseconds.
    pub gpu_time_us: u64,
    /// Number of kernel dispatches.
    pub num_dispatches: usize,
    /// Total data transferred in bytes.
    pub bytes_transferred: usize,
    /// Peak memory usage in bytes.
    pub peak_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability_check() {
        // This just checks the function doesn't panic
        let _ = is_metal_available();
    }

    #[test]
    fn test_metal_config_default() {
        let config = MetalConfig::default();
        assert!(config.max_buffer_size > 0);
        assert!(config.threadgroup_size > 0);
    }
}
