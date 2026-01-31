//! Metal Device Management.
//!
//! This module handles Metal device initialization, buffer management,
//! and shader compilation.

use super::MetalConfig;
use thiserror::Error;

/// Errors from Metal operations.
#[derive(Error, Debug)]
pub enum MetalError {
    #[error("Metal is not available on this system")]
    NotAvailable,

    #[error("Failed to create Metal device")]
    DeviceCreationFailed,

    #[error("Failed to create command queue")]
    CommandQueueCreationFailed,

    #[error("Failed to compile shader: {0}")]
    ShaderCompilationFailed(String),

    #[error("Failed to create buffer: {0}")]
    BufferCreationFailed(String),

    #[error("Buffer size exceeds maximum: {size} > {max}")]
    BufferTooLarge { size: usize, max: usize },

    #[error("Kernel execution failed: {0}")]
    KernelExecutionFailed(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Timeout waiting for GPU")]
    Timeout,
}

/// Result type for Metal operations.
pub type MetalResult<T> = Result<T, MetalError>;

/// Information about a Metal device.
#[derive(Debug, Clone)]
pub struct MetalDeviceInfo {
    /// Device name.
    pub name: String,
    /// Whether the device is low power (integrated GPU).
    pub is_low_power: bool,
    /// Whether the device is headless (no display).
    pub is_headless: bool,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Maximum threadgroup memory size.
    pub max_threadgroup_memory: usize,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: usize,
    /// Recommended working set size in bytes.
    pub recommended_working_set_size: usize,
}

impl Default for MetalDeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            is_low_power: false,
            is_headless: false,
            max_buffer_size: 256 * 1024 * 1024,
            max_threadgroup_memory: 32 * 1024,
            max_threads_per_threadgroup: 1024,
            recommended_working_set_size: 1024 * 1024 * 1024,
        }
    }
}

/// Metal device wrapper.
///
/// On non-macOS systems, this is a stub that returns errors.
pub struct MetalDevice {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: metal_rs::Device,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    command_queue: metal_rs::CommandQueue,
    info: MetalDeviceInfo,
    config: MetalConfig,
}

impl MetalDevice {
    /// Checks if Metal is available on this system.
    pub fn is_available() -> bool {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            metal_rs::Device::system_default().is_some()
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            false
        }
    }

    /// Creates a new Metal device with default configuration.
    pub fn new() -> MetalResult<Self> {
        Self::with_config(MetalConfig::default())
    }

    /// Creates a new Metal device with custom configuration.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn with_config(config: MetalConfig) -> MetalResult<Self> {
        let device = metal_rs::Device::system_default()
            .ok_or(MetalError::DeviceCreationFailed)?;

        let command_queue = device.new_command_queue();

        let info = MetalDeviceInfo {
            name: device.name().to_string(),
            is_low_power: device.is_low_power(),
            is_headless: device.is_headless(),
            max_buffer_size: device.max_buffer_length() as usize,
            max_threadgroup_memory: device.max_threadgroup_memory_length() as usize,
            max_threads_per_threadgroup: device.max_threads_per_threadgroup().width as usize,
            recommended_working_set_size: device.recommended_max_working_set_size() as usize,
        };

        Ok(Self {
            device,
            command_queue,
            info,
            config,
        })
    }

    /// Creates a new Metal device with custom configuration (stub for non-macOS).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn with_config(_config: MetalConfig) -> MetalResult<Self> {
        Err(MetalError::NotAvailable)
    }

    /// Returns device information.
    pub fn info(&self) -> MetalDeviceInfo {
        self.info.clone()
    }

    /// Returns the configuration.
    pub fn config(&self) -> &MetalConfig {
        &self.config
    }

    /// Creates a buffer of the specified size.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn create_buffer(&self, size: usize) -> MetalResult<metal_rs::Buffer> {
        if size > self.config.max_buffer_size {
            return Err(MetalError::BufferTooLarge {
                size,
                max: self.config.max_buffer_size,
            });
        }

        let buffer = self.device.new_buffer(
            size as u64,
            metal_rs::MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Creates a buffer from data.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn create_buffer_with_data<T: Copy>(&self, data: &[T]) -> MetalResult<metal_rs::Buffer> {
        let size = data.len() * std::mem::size_of::<T>();
        if size > self.config.max_buffer_size {
            return Err(MetalError::BufferTooLarge {
                size,
                max: self.config.max_buffer_size,
            });
        }

        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size as u64,
            metal_rs::MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Compiles a compute shader from source.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn compile_shader(&self, source: &str, function_name: &str) -> MetalResult<metal_rs::ComputePipelineState> {
        let options = metal_rs::CompileOptions::new();

        let library = self.device
            .new_library_with_source(source, &options)
            .map_err(|e| MetalError::ShaderCompilationFailed(e.to_string()))?;

        let function = library
            .get_function(function_name, None)
            .map_err(|e| MetalError::ShaderCompilationFailed(e.to_string()))?;

        let pipeline = self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::ShaderCompilationFailed(e.to_string()))?;

        Ok(pipeline)
    }

    /// Creates a new command buffer.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new_command_buffer(&self) -> metal_rs::CommandBuffer {
        self.command_queue.new_command_buffer().to_owned()
    }

    /// Returns the underlying Metal device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn metal_device(&self) -> &metal_rs::Device {
        &self.device
    }
}

/// Buffer pool for reusing Metal buffers.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct BufferPool {
    device: std::sync::Arc<MetalDevice>,
    buffers: std::collections::HashMap<usize, Vec<metal_rs::Buffer>>,
    max_cached: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BufferPool {
    /// Creates a new buffer pool.
    pub fn new(device: std::sync::Arc<MetalDevice>, max_cached: usize) -> Self {
        Self {
            device,
            buffers: std::collections::HashMap::new(),
            max_cached,
        }
    }

    /// Gets or creates a buffer of the specified size.
    pub fn get(&mut self, size: usize) -> MetalResult<metal_rs::Buffer> {
        // Round up to power of 2 for better reuse
        let bucket_size = size.next_power_of_two();

        if let Some(buffers) = self.buffers.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }

        self.device.create_buffer(bucket_size)
    }

    /// Returns a buffer to the pool for reuse.
    pub fn release(&mut self, buffer: metal_rs::Buffer) {
        let size = buffer.length() as usize;
        let entry = self.buffers.entry(size).or_insert_with(Vec::new);

        if entry.len() < self.max_cached {
            entry.push(buffer);
        }
        // Otherwise, buffer is dropped
    }

    /// Clears all cached buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Stub buffer pool for non-macOS.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct BufferPool;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl BufferPool {
    pub fn new(_device: std::sync::Arc<MetalDevice>, _max_cached: usize) -> Self {
        Self
    }

    pub fn clear(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_default() {
        let info = MetalDeviceInfo::default();
        assert!(!info.name.is_empty());
        assert!(info.max_buffer_size > 0);
    }

    #[test]
    fn test_metal_availability() {
        // Just check it doesn't panic
        let available = MetalDevice::is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_device_creation() {
        if MetalDevice::is_available() {
            let device = MetalDevice::new();
            assert!(device.is_ok());

            let device = device.unwrap();
            let info = device.info();
            println!("Metal device: {}", info.name);
        }
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_device_creation_unavailable() {
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::NotAvailable)));
    }
}
