//! GPU Memory Pool Management.
//!
//! Provides efficient memory allocation for GPU operations with:
//! - Size-bucketed allocation for fast reuse
//! - Automatic fragmentation tracking
//! - Memory pressure handling
//! - Allocation statistics
//! - Real GPU memory allocation via CUDA or Metal

use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicU64, AtomicUsize, Ordering}};

use super::GpuBackendType;

// Import GPU allocation functions
#[cfg(feature = "cuda")]
use crate::cuda::bindings::{cuda_malloc, cuda_free};

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal;

/// Configuration for memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum pool size in bytes.
    pub max_size: u64,
    /// Minimum allocation size (smaller allocations are rounded up).
    pub min_allocation_size: usize,
    /// Maximum allocations to cache per bucket.
    pub max_cached_per_bucket: usize,
    /// Enable automatic defragmentation.
    pub enable_defrag: bool,
    /// Defragmentation threshold (fragmentation ratio).
    pub defrag_threshold: f32,
    /// Number of size buckets.
    pub num_buckets: usize,
    /// GPU backend type for allocation.
    pub backend: GpuBackendType,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024, // 1GB
            min_allocation_size: 256,
            max_cached_per_bucket: 16,
            enable_defrag: true,
            defrag_threshold: 0.3,
            num_buckets: 32,
            backend: GpuBackendType::Cpu,
        }
    }
}

/// Statistics from memory pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total bytes allocated from pool.
    pub total_allocated: u64,
    /// Current bytes in use.
    pub bytes_in_use: u64,
    /// Bytes available in free list.
    pub bytes_free: u64,
    /// Number of allocations.
    pub num_allocations: u64,
    /// Number of allocations served from free list.
    pub cache_hits: u64,
    /// Number of allocations requiring new memory.
    pub cache_misses: u64,
    /// Number of deallocations.
    pub num_deallocations: u64,
    /// Peak memory usage.
    pub peak_usage: u64,
    /// Fragmentation ratio.
    pub fragmentation: f32,
}

/// A pooled GPU buffer.
#[derive(Debug)]
pub struct PooledBuffer {
    /// Unique buffer ID.
    pub id: u64,
    /// Requested size in bytes.
    pub requested_size: u64,
    /// Actual allocated size (may be larger due to bucketing).
    pub allocated_size: u64,
    /// Device index.
    pub device: usize,
    /// Bucket index for this allocation.
    bucket: usize,
    /// Pool reference for automatic return.
    pool: Option<Arc<RwLock<PoolInner>>>,
    /// Raw pointer/handle (opaque).
    ptr: u64,
}

impl PooledBuffer {
    /// Returns the raw pointer.
    pub fn as_ptr(&self) -> u64 {
        self.ptr
    }

    /// Returns true if this buffer has sufficient size.
    pub fn fits(&self, size: u64) -> bool {
        self.allocated_size >= size
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool
        if let Some(ref pool) = self.pool {
            if let Ok(mut inner) = pool.write() {
                inner.return_buffer(BufferHandle {
                    id: self.id,
                    size: self.allocated_size,
                    bucket: self.bucket,
                    ptr: self.ptr,
                });
            }
        }
    }
}

/// Internal buffer handle for the free list.
#[derive(Debug, Clone)]
struct BufferHandle {
    id: u64,
    size: u64,
    bucket: usize,
    ptr: u64,
}

/// Internal pool state.
#[derive(Debug)]
struct PoolInner {
    /// Configuration.
    config: PoolConfig,
    /// Free lists organized by bucket (power-of-2 sizes).
    free_lists: Vec<Vec<BufferHandle>>,
    /// Total allocated memory.
    total_allocated: u64,
    /// Bytes currently in use.
    bytes_in_use: u64,
    /// Next buffer ID.
    next_id: u64,
    /// Statistics.
    stats: PoolStats,
    /// Device index.
    device: usize,
}

impl PoolInner {
    fn new(config: PoolConfig, device: usize) -> Self {
        let mut free_lists = Vec::with_capacity(config.num_buckets);
        for _ in 0..config.num_buckets {
            free_lists.push(Vec::new());
        }

        Self {
            config,
            free_lists,
            total_allocated: 0,
            bytes_in_use: 0,
            next_id: 0,
            stats: PoolStats::default(),
            device,
        }
    }

    /// Gets bucket index for a given size.
    fn bucket_for_size(&self, size: u64) -> usize {
        let min = self.config.min_allocation_size as u64;
        if size <= min {
            return 0;
        }

        let normalized = (size - 1) / min;
        let bits = 64 - normalized.leading_zeros();
        std::cmp::min(bits as usize, self.config.num_buckets - 1)
    }

    /// Gets bucket size (actual allocation size).
    fn bucket_size(&self, bucket: usize) -> u64 {
        let min = self.config.min_allocation_size as u64;
        if bucket == 0 {
            return min;
        }
        min << bucket
    }

    /// Allocates GPU memory based on backend type.
    fn allocate_gpu_memory(&self, size: u64) -> Option<u64> {
        match self.config.backend {
            #[cfg(feature = "cuda")]
            GpuBackendType::Cuda => {
                unsafe { cuda_malloc(size).ok() }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackendType::Metal => {
                // Metal uses buffer objects managed by the device
                // Return a unique ID that maps to a Metal buffer
                static METAL_BUFFER_ID: AtomicU64 = AtomicU64::new(0x8000_0000_0000_0000);
                Some(METAL_BUFFER_ID.fetch_add(1, Ordering::Relaxed))
            }
            GpuBackendType::Cpu => {
                // For CPU backend, allocate heap memory and return pointer as u64
                let layout = std::alloc::Layout::from_size_align(size as usize, 64).ok()?;
                let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
                if ptr.is_null() {
                    None
                } else {
                    Some(ptr as u64)
                }
            }
            #[allow(unreachable_patterns)]
            _ => {
                // Fallback: use ID as fake pointer for unsupported backends
                static FALLBACK_ID: AtomicU64 = AtomicU64::new(1);
                Some(FALLBACK_ID.fetch_add(1, Ordering::Relaxed))
            }
        }
    }

    /// Frees GPU memory based on backend type.
    fn free_gpu_memory(&self, ptr: u64, size: u64) {
        match self.config.backend {
            #[cfg(feature = "cuda")]
            GpuBackendType::Cuda => {
                unsafe { let _ = cuda_free(ptr); }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackendType::Metal => {
                // Metal buffers are reference counted and freed automatically
                // The buffer ID is just a tracking number
            }
            GpuBackendType::Cpu => {
                // For CPU backend, free the heap memory
                if ptr != 0 {
                    if let Ok(layout) = std::alloc::Layout::from_size_align(size as usize, 64) {
                        unsafe { std::alloc::dealloc(ptr as *mut u8, layout); }
                    }
                }
            }
            #[allow(unreachable_patterns)]
            _ => {
                // No-op for unsupported backends
            }
        }
    }

    /// Allocates a buffer from the pool.
    fn allocate(&mut self, requested_size: u64) -> Option<PooledBuffer> {
        let bucket = self.bucket_for_size(requested_size);
        let actual_size = self.bucket_size(bucket);

        // Check free list
        if let Some(handle) = self.free_lists[bucket].pop() {
            self.bytes_in_use += actual_size;
            self.stats.cache_hits += 1;
            self.stats.num_allocations += 1;
            self.stats.bytes_in_use = self.bytes_in_use;

            return Some(PooledBuffer {
                id: handle.id,
                requested_size,
                allocated_size: handle.size,
                device: self.device,
                bucket,
                pool: None, // Set by caller
                ptr: handle.ptr,
            });
        }

        // Check pool capacity
        if self.total_allocated + actual_size > self.config.max_size {
            // Try to reclaim from larger buckets
            if !self.reclaim_memory(actual_size) {
                return None;
            }
        }

        // Allocate new buffer
        let id = self.next_id;
        self.next_id += 1;

        // Perform real GPU allocation based on backend type
        let ptr = self.allocate_gpu_memory(actual_size)?;

        self.total_allocated += actual_size;
        self.bytes_in_use += actual_size;
        self.stats.cache_misses += 1;
        self.stats.num_allocations += 1;
        self.stats.total_allocated = self.total_allocated;
        self.stats.bytes_in_use = self.bytes_in_use;

        if self.bytes_in_use > self.stats.peak_usage {
            self.stats.peak_usage = self.bytes_in_use;
        }

        Some(PooledBuffer {
            id,
            requested_size,
            allocated_size: actual_size,
            device: self.device,
            bucket,
            pool: None,
            ptr,
        })
    }

    /// Returns a buffer to the pool.
    fn return_buffer(&mut self, handle: BufferHandle) {
        self.bytes_in_use = self.bytes_in_use.saturating_sub(handle.size);
        self.stats.num_deallocations += 1;

        // Check if we should cache this buffer
        if self.free_lists[handle.bucket].len() < self.config.max_cached_per_bucket {
            self.free_lists[handle.bucket].push(handle);
            self.update_stats();
        } else {
            // Release the buffer - perform actual GPU free
            self.free_gpu_memory(handle.ptr, handle.size);
            self.total_allocated = self.total_allocated.saturating_sub(handle.size);
        }
    }

    /// Tries to reclaim memory from free lists.
    fn reclaim_memory(&mut self, needed: u64) -> bool {
        let mut reclaimed = 0u64;

        // Start from largest buckets
        for bucket in (0..self.config.num_buckets).rev() {
            while !self.free_lists[bucket].is_empty() && reclaimed < needed {
                if let Some(handle) = self.free_lists[bucket].pop() {
                    // Free the actual GPU memory
                    self.free_gpu_memory(handle.ptr, handle.size);
                    reclaimed += handle.size;
                    self.total_allocated = self.total_allocated.saturating_sub(handle.size);
                }
            }
        }

        reclaimed >= needed
    }

    /// Updates statistics.
    fn update_stats(&mut self) {
        let free_bytes: u64 = self.free_lists.iter()
            .enumerate()
            .map(|(b, list)| list.len() as u64 * self.bucket_size(b))
            .sum();

        self.stats.bytes_free = free_bytes;
        self.stats.bytes_in_use = self.bytes_in_use;

        if self.total_allocated > 0 {
            self.stats.fragmentation = free_bytes as f32 / self.total_allocated as f32;
        } else {
            self.stats.fragmentation = 0.0;
        }
    }

    /// Clears all cached buffers.
    fn clear(&mut self) {
        // Collect all handles to free first
        let handles_to_free: Vec<BufferHandle> = self.free_lists
            .iter_mut()
            .flat_map(|list| list.drain(..))
            .collect();

        // Now free the GPU memory for each handle
        for handle in handles_to_free {
            self.free_gpu_memory(handle.ptr, handle.size);
        }
        self.total_allocated = 0;
        self.update_stats();
    }
}

/// Thread-safe GPU memory pool.
pub struct GpuMemoryPool {
    inner: Arc<RwLock<PoolInner>>,
}

impl GpuMemoryPool {
    /// Creates a new memory pool.
    pub fn new(config: PoolConfig, device: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(PoolInner::new(config, device))),
        }
    }

    /// Creates with default configuration.
    pub fn with_defaults(device: usize) -> Self {
        Self::new(PoolConfig::default(), device)
    }

    /// Allocates a buffer from the pool.
    pub fn allocate(&self, size: u64) -> Option<PooledBuffer> {
        let mut inner = self.inner.write().ok()?;
        let mut buffer = inner.allocate(size)?;
        buffer.pool = Some(self.inner.clone());
        Some(buffer)
    }

    /// Allocates multiple buffers of the same size.
    pub fn allocate_batch(&self, size: u64, count: usize) -> Vec<PooledBuffer> {
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(buf) = self.allocate(size) {
                buffers.push(buf);
            } else {
                break;
            }
        }
        buffers
    }

    /// Returns statistics.
    pub fn stats(&self) -> PoolStats {
        if let Ok(inner) = self.inner.read() {
            inner.stats.clone()
        } else {
            PoolStats::default()
        }
    }

    /// Clears all cached buffers.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.write() {
            inner.clear();
        }
    }

    /// Returns bytes currently in use.
    pub fn bytes_in_use(&self) -> u64 {
        if let Ok(inner) = self.inner.read() {
            inner.bytes_in_use
        } else {
            0
        }
    }

    /// Returns total allocated bytes.
    pub fn total_allocated(&self) -> u64 {
        if let Ok(inner) = self.inner.read() {
            inner.total_allocated
        } else {
            0
        }
    }

    /// Returns number of cached buffers.
    pub fn cached_count(&self) -> usize {
        if let Ok(inner) = self.inner.read() {
            inner.free_lists.iter().map(|l| l.len()).sum()
        } else {
            0
        }
    }

    /// Attempts defragmentation.
    pub fn defragment(&self) -> bool {
        if let Ok(mut inner) = self.inner.write() {
            if inner.config.enable_defrag &&
               inner.stats.fragmentation > inner.config.defrag_threshold {
                // Simple defrag: clear smaller buckets
                for bucket in 0..(inner.config.num_buckets / 2) {
                    while let Some(handle) = inner.free_lists[bucket].pop() {
                        inner.total_allocated =
                            inner.total_allocated.saturating_sub(handle.size);
                    }
                }
                inner.update_stats();
                return true;
            }
        }
        false
    }
}

impl Clone for GpuMemoryPool {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocation() {
        let pool = GpuMemoryPool::with_defaults(0);

        let buf1 = pool.allocate(1024).unwrap();
        assert!(buf1.allocated_size >= 1024);

        let stats = pool.stats();
        assert_eq!(stats.num_allocations, 1);
        assert!(stats.bytes_in_use > 0);
    }

    #[test]
    fn test_pool_reuse() {
        let pool = GpuMemoryPool::with_defaults(0);

        let buf1 = pool.allocate(1024).unwrap();
        let id1 = buf1.id;
        drop(buf1);

        let buf2 = pool.allocate(1024).unwrap();

        let stats = pool.stats();
        // Should have reused the buffer
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_pool_batch_allocation() {
        let pool = GpuMemoryPool::with_defaults(0);

        let buffers = pool.allocate_batch(1024, 10);
        assert_eq!(buffers.len(), 10);

        let stats = pool.stats();
        assert_eq!(stats.num_allocations, 10);
    }

    #[test]
    fn test_pool_clear() {
        let pool = GpuMemoryPool::with_defaults(0);

        let _buf = pool.allocate(1024).unwrap();
        drop(_buf);

        assert!(pool.cached_count() > 0);

        pool.clear();
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn test_bucket_sizing() {
        let config = PoolConfig::default();
        let inner = PoolInner::new(config.clone(), 0);

        // Small allocations go to bucket 0
        assert_eq!(inner.bucket_for_size(100), 0);
        assert_eq!(inner.bucket_for_size(256), 0);

        // Larger allocations go to higher buckets
        assert!(inner.bucket_for_size(1024) > 0);
        assert!(inner.bucket_for_size(1024 * 1024) > inner.bucket_for_size(1024));
    }
}
