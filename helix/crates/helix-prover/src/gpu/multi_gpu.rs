//! Multi-GPU Load Balancing.
//!
//! Provides infrastructure for distributing work across multiple GPUs with:
//! - Automatic device discovery and capability detection
//! - Multiple load balancing strategies
//! - Work partitioning and result aggregation
//! - Cross-device synchronization

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{GpuBackendType, GpuConfig, pool::GpuMemoryPool, async_ops::AsyncOpQueue};

/// Information about a GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index.
    pub index: usize,
    /// Backend type.
    pub backend: GpuBackendType,
    /// Device name.
    pub name: String,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Available memory in bytes.
    pub available_memory: u64,
    /// Compute capability or feature level.
    pub compute_capability: (u32, u32),
    /// Number of compute units (SMs/CUs).
    pub compute_units: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Relative performance score (higher = faster).
    pub performance_score: f32,
    /// Whether the device is currently available.
    pub available: bool,
}

impl DeviceInfo {
    /// Creates device info for CPU.
    pub fn cpu() -> Self {
        let num_cores = num_cpus::get() as u32;
        Self {
            index: 0,
            backend: GpuBackendType::Cpu,
            name: "CPU".to_string(),
            total_memory: 0, // Not tracked for CPU
            available_memory: 0,
            compute_capability: (0, 0),
            compute_units: num_cores,
            max_threads_per_block: 1,
            performance_score: 1.0, // Baseline
            available: true,
        }
    }
}

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Round-robin distribution.
    RoundRobin,
    /// Weighted by device performance.
    Weighted,
    /// Least loaded device.
    LeastLoaded,
    /// Memory-aware (prefer devices with more free memory).
    MemoryAware,
    /// Static partitioning (equal work per device).
    Static,
}

impl Default for LoadBalanceStrategy {
    fn default() -> Self {
        LoadBalanceStrategy::Weighted
    }
}

/// Device selector for choosing which device to use.
pub struct DeviceSelector {
    devices: Vec<DeviceInfo>,
    strategy: LoadBalanceStrategy,
    /// Current device for round-robin.
    current_device: AtomicU64,
    /// Per-device load counters.
    device_loads: RwLock<Vec<u64>>,
    /// Per-device memory usage.
    memory_usage: RwLock<Vec<u64>>,
}

impl DeviceSelector {
    /// Creates a new device selector.
    pub fn new(devices: Vec<DeviceInfo>, strategy: LoadBalanceStrategy) -> Self {
        let num_devices = devices.len();
        Self {
            devices,
            strategy,
            current_device: AtomicU64::new(0),
            device_loads: RwLock::new(vec![0; num_devices]),
            memory_usage: RwLock::new(vec![0; num_devices]),
        }
    }

    /// Selects a device for work.
    pub fn select(&self) -> usize {
        if self.devices.is_empty() {
            return 0;
        }

        match self.strategy {
            LoadBalanceStrategy::RoundRobin => self.select_round_robin(),
            LoadBalanceStrategy::Weighted => self.select_weighted(),
            LoadBalanceStrategy::LeastLoaded => self.select_least_loaded(),
            LoadBalanceStrategy::MemoryAware => self.select_memory_aware(),
            LoadBalanceStrategy::Static => 0, // Static uses all devices equally
        }
    }

    /// Selects device using round-robin.
    fn select_round_robin(&self) -> usize {
        let current = self.current_device.fetch_add(1, Ordering::Relaxed);
        (current as usize) % self.devices.len()
    }

    /// Selects device weighted by performance.
    fn select_weighted(&self) -> usize {
        let loads = self.device_loads.read().unwrap();
        let total_score: f32 = self.devices.iter().map(|d| d.performance_score).sum();

        let mut best_device = 0;
        let mut best_ratio = f32::MAX;

        for (i, device) in self.devices.iter().enumerate() {
            // Compute load ratio (lower is better)
            let expected_share = device.performance_score / total_score;
            let actual_share = if loads.iter().sum::<u64>() > 0 {
                loads[i] as f32 / loads.iter().sum::<u64>() as f32
            } else {
                0.0
            };

            let ratio = actual_share / expected_share.max(0.001);
            if ratio < best_ratio {
                best_ratio = ratio;
                best_device = i;
            }
        }

        best_device
    }

    /// Selects least loaded device.
    fn select_least_loaded(&self) -> usize {
        let loads = self.device_loads.read().unwrap();
        loads.iter()
            .enumerate()
            .min_by_key(|(_, load)| *load)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Selects device with most available memory.
    fn select_memory_aware(&self) -> usize {
        let usage = self.memory_usage.read().unwrap();

        self.devices.iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                let avail_a = a.available_memory.saturating_sub(usage[*i]);
                let avail_b = b.available_memory.saturating_sub(usage[*j]);
                avail_a.cmp(&avail_b)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Records work assignment to a device.
    pub fn record_assignment(&self, device: usize, work_units: u64, memory: u64) {
        if let Ok(mut loads) = self.device_loads.write() {
            if device < loads.len() {
                loads[device] += work_units;
            }
        }
        if let Ok(mut usage) = self.memory_usage.write() {
            if device < usage.len() {
                usage[device] += memory;
            }
        }
    }

    /// Records work completion on a device.
    pub fn record_completion(&self, device: usize, work_units: u64, memory: u64) {
        if let Ok(mut loads) = self.device_loads.write() {
            if device < loads.len() {
                loads[device] = loads[device].saturating_sub(work_units);
            }
        }
        if let Ok(mut usage) = self.memory_usage.write() {
            if device < usage.len() {
                usage[device] = usage[device].saturating_sub(memory);
            }
        }
    }

    /// Returns available devices.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Returns number of devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Returns current loads.
    pub fn loads(&self) -> Vec<u64> {
        self.device_loads.read().unwrap().clone()
    }
}

/// Work partitioning for multi-GPU execution.
pub struct WorkDistributor {
    selector: Arc<DeviceSelector>,
    strategy: LoadBalanceStrategy,
}

/// A partition of work for a single device.
#[derive(Debug, Clone)]
pub struct WorkPartition {
    /// Device index.
    pub device: usize,
    /// Start index in the input.
    pub start: usize,
    /// Number of elements.
    pub count: usize,
    /// Estimated memory requirement.
    pub memory_estimate: u64,
}

impl WorkDistributor {
    /// Creates a new work distributor.
    pub fn new(selector: Arc<DeviceSelector>, strategy: LoadBalanceStrategy) -> Self {
        Self { selector, strategy }
    }

    /// Partitions work across devices.
    pub fn partition(&self, total_elements: usize, element_size: usize) -> Vec<WorkPartition> {
        let num_devices = self.selector.num_devices();
        if num_devices == 0 || total_elements == 0 {
            return vec![];
        }

        match self.strategy {
            LoadBalanceStrategy::Static => self.partition_static(total_elements, element_size),
            LoadBalanceStrategy::Weighted => self.partition_weighted(total_elements, element_size),
            LoadBalanceStrategy::MemoryAware => self.partition_memory_aware(total_elements, element_size),
            _ => self.partition_static(total_elements, element_size),
        }
    }

    /// Static partitioning (equal work per device).
    fn partition_static(&self, total_elements: usize, element_size: usize) -> Vec<WorkPartition> {
        let num_devices = self.selector.num_devices();
        let base_count = total_elements / num_devices;
        let remainder = total_elements % num_devices;

        let mut partitions = Vec::with_capacity(num_devices);
        let mut offset = 0;

        for i in 0..num_devices {
            let count = base_count + if i < remainder { 1 } else { 0 };
            let memory = (count * element_size) as u64;

            partitions.push(WorkPartition {
                device: i,
                start: offset,
                count,
                memory_estimate: memory,
            });

            offset += count;
        }

        partitions
    }

    /// Weighted partitioning by device performance.
    fn partition_weighted(&self, total_elements: usize, element_size: usize) -> Vec<WorkPartition> {
        let devices = self.selector.devices();
        let total_score: f32 = devices.iter().map(|d| d.performance_score).sum();

        let mut partitions = Vec::with_capacity(devices.len());
        let mut offset = 0;
        let mut remaining = total_elements;

        for (i, device) in devices.iter().enumerate() {
            let share = device.performance_score / total_score;
            let count = if i == devices.len() - 1 {
                remaining // Give rest to last device
            } else {
                ((total_elements as f32) * share).round() as usize
            };

            let count = count.min(remaining);
            let memory = (count * element_size) as u64;

            partitions.push(WorkPartition {
                device: i,
                start: offset,
                count,
                memory_estimate: memory,
            });

            offset += count;
            remaining -= count;
        }

        partitions
    }

    /// Memory-aware partitioning.
    fn partition_memory_aware(&self, total_elements: usize, element_size: usize) -> Vec<WorkPartition> {
        let devices = self.selector.devices();
        let memory_usage = self.selector.memory_usage.read().unwrap();

        // Calculate available memory per device
        let available: Vec<u64> = devices.iter()
            .enumerate()
            .map(|(i, d)| d.available_memory.saturating_sub(memory_usage[i]))
            .collect();

        let total_available: u64 = available.iter().sum();
        if total_available == 0 {
            return self.partition_static(total_elements, element_size);
        }

        let mut partitions = Vec::with_capacity(devices.len());
        let mut offset = 0;
        let mut remaining = total_elements;

        for (i, &avail) in available.iter().enumerate() {
            let share = avail as f64 / total_available as f64;
            let count = if i == devices.len() - 1 {
                remaining
            } else {
                ((total_elements as f64) * share).round() as usize
            };

            let count = count.min(remaining);
            let memory = (count * element_size) as u64;

            partitions.push(WorkPartition {
                device: i,
                start: offset,
                count,
                memory_estimate: memory,
            });

            offset += count;
            remaining -= count;
        }

        partitions
    }

    /// Selects a single device for the work.
    pub fn select_single(&self) -> usize {
        self.selector.select()
    }
}

/// Multi-GPU manager for coordinating work across devices.
pub struct MultiGpuManager {
    /// Device information.
    devices: Vec<DeviceInfo>,
    /// Device selector.
    selector: Arc<DeviceSelector>,
    /// Work distributor.
    distributor: WorkDistributor,
    /// Memory pools per device.
    pools: Vec<GpuMemoryPool>,
    /// Async queues per device.
    queues: Vec<AsyncOpQueue>,
    /// Statistics.
    stats: RwLock<MultiGpuStats>,
}

/// Statistics for multi-GPU operations.
#[derive(Debug, Clone, Default)]
pub struct MultiGpuStats {
    /// Operations per device.
    pub ops_per_device: Vec<u64>,
    /// Bytes processed per device.
    pub bytes_per_device: Vec<u64>,
    /// Time spent per device (microseconds).
    pub time_per_device: Vec<u64>,
    /// Total operations.
    pub total_ops: u64,
    /// Total bytes processed.
    pub total_bytes: u64,
    /// Cross-device transfers.
    pub cross_device_transfers: u64,
}

impl MultiGpuManager {
    /// Creates a new multi-GPU manager.
    pub fn new(config: &GpuConfig) -> Self {
        let devices = Self::discover_devices(config);
        let num_devices = devices.len();

        let selector = Arc::new(DeviceSelector::new(
            devices.clone(),
            LoadBalanceStrategy::Weighted,
        ));

        let distributor = WorkDistributor::new(
            selector.clone(),
            LoadBalanceStrategy::Weighted,
        );

        // Create pools and queues for each device
        let pools: Vec<_> = (0..num_devices)
            .map(|i| GpuMemoryPool::with_defaults(i))
            .collect();

        let queues: Vec<_> = (0..num_devices)
            .map(|_| AsyncOpQueue::new(config.num_streams))
            .collect();

        Self {
            devices,
            selector,
            distributor,
            pools,
            queues,
            stats: RwLock::new(MultiGpuStats {
                ops_per_device: vec![0; num_devices],
                bytes_per_device: vec![0; num_devices],
                time_per_device: vec![0; num_devices],
                ..Default::default()
            }),
        }
    }

    /// Discovers available GPU devices.
    fn discover_devices(config: &GpuConfig) -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        // Check CUDA devices
        #[cfg(feature = "cuda")]
        {
            for cuda_dev in crate::cuda::enumerate_devices() {
                devices.push(DeviceInfo {
                    index: cuda_dev.index,
                    backend: GpuBackendType::Cuda,
                    name: cuda_dev.name,
                    total_memory: cuda_dev.total_memory,
                    available_memory: cuda_dev.total_memory, // Would query actual available
                    compute_capability: cuda_dev.compute_capability,
                    compute_units: cuda_dev.sm_count,
                    max_threads_per_block: cuda_dev.max_threads_per_block,
                    performance_score: (cuda_dev.sm_count as f32) *
                                      (cuda_dev.compute_capability.0 as f32 + 0.1 * cuda_dev.compute_capability.1 as f32),
                    available: true,
                });
            }
        }

        // Check Metal device
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(metal_info) = crate::metal::get_device_info() {
                devices.push(DeviceInfo {
                    index: 0,
                    backend: GpuBackendType::Metal,
                    name: metal_info.name,
                    total_memory: metal_info.recommended_working_set_size as u64,
                    available_memory: metal_info.recommended_working_set_size as u64,
                    compute_capability: (0, 0),
                    compute_units: 1, // Metal doesn't expose this
                    max_threads_per_block: metal_info.max_threads_per_threadgroup as u32,
                    performance_score: 10.0, // Estimated
                    available: true,
                });
            }
        }

        // Add CPU as fallback
        if config.enable_cpu_fallback || devices.is_empty() {
            devices.push(DeviceInfo::cpu());
        }

        // Sort by performance (highest first)
        devices.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap());

        devices
    }

    /// Returns number of devices.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Returns device information.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Gets memory pool for a device.
    pub fn pool(&self, device: usize) -> &GpuMemoryPool {
        &self.pools[device.min(self.pools.len() - 1)]
    }

    /// Gets async queue for a device.
    pub fn queue(&self, device: usize) -> &AsyncOpQueue {
        &self.queues[device.min(self.queues.len() - 1)]
    }

    /// Selects best device for work.
    pub fn select_device(&self) -> usize {
        self.selector.select()
    }

    /// Partitions work across devices.
    pub fn partition_work(&self, total_elements: usize, element_size: usize) -> Vec<WorkPartition> {
        self.distributor.partition(total_elements, element_size)
    }

    /// Records operation start.
    pub fn record_start(&self, device: usize, bytes: u64) {
        self.selector.record_assignment(device, 1, bytes);
    }

    /// Records operation completion.
    pub fn record_complete(&self, device: usize, bytes: u64, time_us: u64) {
        self.selector.record_completion(device, 1, bytes);

        if let Ok(mut stats) = self.stats.write() {
            if device < stats.ops_per_device.len() {
                stats.ops_per_device[device] += 1;
                stats.bytes_per_device[device] += bytes;
                stats.time_per_device[device] += time_us;
            }
            stats.total_ops += 1;
            stats.total_bytes += bytes;
        }
    }

    /// Synchronizes all devices.
    pub fn sync_all(&self) {
        for queue in &self.queues {
            queue.sync_all();
        }
    }

    /// Returns statistics.
    pub fn stats(&self) -> MultiGpuStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Returns best backend type available.
    pub fn best_backend(&self) -> GpuBackendType {
        self.devices.first()
            .map(|d| d.backend)
            .unwrap_or(GpuBackendType::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selector_round_robin() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo::cpu(),
        ];
        let selector = DeviceSelector::new(devices, LoadBalanceStrategy::RoundRobin);

        assert_eq!(selector.select(), 0);
        assert_eq!(selector.select(), 1);
        assert_eq!(selector.select(), 0);
    }

    #[test]
    fn test_work_partitioning() {
        let devices = vec![
            DeviceInfo {
                performance_score: 10.0,
                ..DeviceInfo::cpu()
            },
            DeviceInfo {
                performance_score: 5.0,
                ..DeviceInfo::cpu()
            },
        ];

        let selector = Arc::new(DeviceSelector::new(devices, LoadBalanceStrategy::Weighted));
        let distributor = WorkDistributor::new(selector, LoadBalanceStrategy::Static);

        let partitions = distributor.partition(100, 4);
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].count + partitions[1].count, 100);
    }

    #[test]
    fn test_weighted_partitioning() {
        let devices = vec![
            DeviceInfo {
                performance_score: 20.0,
                ..DeviceInfo::cpu()
            },
            DeviceInfo {
                performance_score: 10.0,
                ..DeviceInfo::cpu()
            },
        ];

        let selector = Arc::new(DeviceSelector::new(devices, LoadBalanceStrategy::Weighted));
        let distributor = WorkDistributor::new(selector, LoadBalanceStrategy::Weighted);

        let partitions = distributor.partition(100, 4);

        // First device should get more work
        assert!(partitions[0].count > partitions[1].count);
    }

    #[test]
    fn test_multi_gpu_manager_creation() {
        let config = GpuConfig::default();
        let manager = MultiGpuManager::new(&config);

        // Should have at least CPU
        assert!(manager.num_devices() > 0);
    }
}
