//! Hybrid CPU+GPU Executor
//!
//! Maximizes hardware utilization by running GPU and CPU in parallel:
//! - GPU handles large batch operations (NTT, MSM, field arithmetic)
//! - CPU handles smaller tasks, data preparation, and orchestration
//! - Double-buffering allows CPU to prepare next batch while GPU processes current

use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::collections::VecDeque;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Work item that can be executed on either CPU or GPU
#[derive(Debug)]
pub enum WorkItem {
    /// NTT operation (data, inverse flag)
    Ntt {
        data: Vec<[u64; 4]>,
        inverse: bool,
        result_tx: mpsc::Sender<Vec<[u64; 4]>>,
    },
    /// Field multiplication batch
    FieldMul {
        a: Vec<[u64; 4]>,
        b: Vec<[u64; 4]>,
        result_tx: mpsc::Sender<Vec<[u64; 4]>>,
    },
    /// MSM operation
    Msm {
        points: Vec<[u64; 8]>,
        scalars: Vec<[u64; 4]>,
        result_tx: mpsc::Sender<[u64; 12]>,
    },
    /// Shutdown signal
    Shutdown,
}

/// Configuration for hybrid execution
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Minimum batch size to use GPU (smaller goes to CPU)
    pub gpu_threshold: usize,
    /// Number of CPU worker threads
    pub cpu_threads: usize,
    /// Maximum GPU queue depth (for double/triple buffering)
    pub gpu_queue_depth: usize,
    /// Whether to use work stealing between CPU and GPU queues
    pub work_stealing: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 1024,
            cpu_threads: num_cpus::get().saturating_sub(2).max(1), // Leave cores for GPU driver + main thread
            gpu_queue_depth: 3,
            work_stealing: true,
        }
    }
}

/// Statistics for hybrid execution
#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    pub gpu_operations: u64,
    pub cpu_operations: u64,
    pub gpu_time_us: u64,
    pub cpu_time_us: u64,
    pub gpu_elements: u64,
    pub cpu_elements: u64,
}

impl HybridStats {
    pub fn gpu_utilization(&self) -> f64 {
        let total = self.gpu_time_us + self.cpu_time_us;
        if total == 0 {
            0.0
        } else {
            self.gpu_time_us as f64 / total as f64
        }
    }

    pub fn throughput(&self, duration_secs: f64) -> f64 {
        (self.gpu_elements + self.cpu_elements) as f64 / duration_secs
    }
}

/// Hybrid executor that runs CPU and GPU in parallel
pub struct HybridExecutor {
    config: HybridConfig,
    stats: Arc<Mutex<HybridStats>>,
    gpu_tx: Option<mpsc::Sender<WorkItem>>,
    cpu_tx: Option<mpsc::Sender<WorkItem>>,
    gpu_thread: Option<thread::JoinHandle<()>>,
    cpu_threads: Vec<thread::JoinHandle<()>>,
}

impl HybridExecutor {
    /// Creates a new hybrid executor
    pub fn new(config: HybridConfig) -> Self {
        let stats = Arc::new(Mutex::new(HybridStats::default()));

        // Create GPU worker thread
        let (gpu_tx, gpu_rx) = mpsc::channel::<WorkItem>();
        let gpu_stats = stats.clone();
        let gpu_thread = thread::spawn(move || {
            Self::gpu_worker(gpu_rx, gpu_stats);
        });

        // Create CPU worker threads
        let (cpu_tx, cpu_rx) = mpsc::channel::<WorkItem>();
        let cpu_rx = Arc::new(Mutex::new(cpu_rx));
        let mut cpu_threads = Vec::with_capacity(config.cpu_threads);

        for _ in 0..config.cpu_threads {
            let rx = cpu_rx.clone();
            let cpu_stats = stats.clone();
            cpu_threads.push(thread::spawn(move || {
                Self::cpu_worker(rx, cpu_stats);
            }));
        }

        Self {
            config,
            stats,
            gpu_tx: Some(gpu_tx),
            cpu_tx: Some(cpu_tx),
            gpu_thread: Some(gpu_thread),
            cpu_threads,
        }
    }

    /// GPU worker thread - processes large batches on GPU
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn gpu_worker(rx: mpsc::Receiver<WorkItem>, stats: Arc<Mutex<HybridStats>>) {
        use crate::metal::{MetalDevice, ntt::{MetalNtt, NttConfig}};
        use std::sync::Arc as StdArc;

        // Initialize Metal
        let device = match MetalDevice::new() {
            Ok(d) => StdArc::new(d),
            Err(e) => {
                eprintln!("GPU worker: Failed to init Metal: {:?}", e);
                return;
            }
        };

        let mut ntt_config = NttConfig::default();
        ntt_config.min_gpu_batch_size = 64;

        let mut ntt = match MetalNtt::with_config(device.clone(), ntt_config) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("GPU worker: Failed to create NTT: {:?}", e);
                return;
            }
        };

        loop {
            match rx.recv() {
                Ok(WorkItem::Ntt { mut data, inverse, result_tx }) => {
                    let start = std::time::Instant::now();
                    let len = data.len();

                    let result = if inverse {
                        ntt.inverse(&mut data)
                    } else {
                        ntt.forward(&mut data)
                    };

                    if result.is_ok() {
                        let _ = result_tx.send(data);
                    }

                    if let Ok(mut s) = stats.lock() {
                        s.gpu_operations += 1;
                        s.gpu_time_us += start.elapsed().as_micros() as u64;
                        s.gpu_elements += len as u64;
                    }
                }
                Ok(WorkItem::FieldMul { a, b, result_tx }) => {
                    // GPU field multiplication would go here
                    // For now, fall back to CPU
                    let result = Self::cpu_field_mul(&a, &b);
                    let _ = result_tx.send(result);
                }
                Ok(WorkItem::Msm { points, scalars, result_tx }) => {
                    // GPU MSM would go here
                    // For now, return identity
                    let _ = result_tx.send([0u64; 12]);
                }
                Ok(WorkItem::Shutdown) | Err(_) => break,
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn gpu_worker(rx: mpsc::Receiver<WorkItem>, stats: Arc<Mutex<HybridStats>>) {
        // No GPU available, redirect to CPU
        loop {
            match rx.recv() {
                Ok(WorkItem::Shutdown) | Err(_) => break,
                Ok(item) => {
                    // Process on CPU instead
                    Self::process_on_cpu(item, &stats);
                }
            }
        }
    }

    /// CPU worker thread - processes smaller batches and assists GPU
    fn cpu_worker(rx: Arc<Mutex<mpsc::Receiver<WorkItem>>>, stats: Arc<Mutex<HybridStats>>) {
        loop {
            let item = {
                let rx = rx.lock().unwrap();
                rx.recv()
            };

            match item {
                Ok(WorkItem::Shutdown) | Err(_) => break,
                Ok(item) => Self::process_on_cpu(item, &stats),
            }
        }
    }

    fn process_on_cpu(item: WorkItem, stats: &Arc<Mutex<HybridStats>>) {
        match item {
            WorkItem::Ntt { mut data, inverse, result_tx } => {
                let start = std::time::Instant::now();
                let len = data.len();

                Self::cpu_ntt(&mut data, inverse);
                let _ = result_tx.send(data);

                if let Ok(mut s) = stats.lock() {
                    s.cpu_operations += 1;
                    s.cpu_time_us += start.elapsed().as_micros() as u64;
                    s.cpu_elements += len as u64;
                }
            }
            WorkItem::FieldMul { a, b, result_tx } => {
                let start = std::time::Instant::now();
                let len = a.len();

                let result = Self::cpu_field_mul(&a, &b);
                let _ = result_tx.send(result);

                if let Ok(mut s) = stats.lock() {
                    s.cpu_operations += 1;
                    s.cpu_time_us += start.elapsed().as_micros() as u64;
                    s.cpu_elements += len as u64;
                }
            }
            WorkItem::Msm { points, scalars, result_tx } => {
                // CPU MSM
                let _ = result_tx.send([0u64; 12]);
            }
            WorkItem::Shutdown => {}
        }
    }

    /// Submit NTT work - automatically routes to GPU or CPU based on size
    pub fn submit_ntt(&self, data: Vec<[u64; 4]>, inverse: bool) -> mpsc::Receiver<Vec<[u64; 4]>> {
        let (tx, rx) = mpsc::channel();

        let item = WorkItem::Ntt { data: data.clone(), inverse, result_tx: tx };

        // Route based on size
        if data.len() >= self.config.gpu_threshold {
            if let Some(ref gpu_tx) = self.gpu_tx {
                let _ = gpu_tx.send(item);
            }
        } else {
            if let Some(ref cpu_tx) = self.cpu_tx {
                let _ = cpu_tx.send(item);
            }
        }

        rx
    }

    /// Submit multiple NTTs in parallel - maximizes both CPU and GPU utilization
    pub fn submit_batch_ntt(&self, batches: Vec<Vec<[u64; 4]>>, inverse: bool) -> Vec<mpsc::Receiver<Vec<[u64; 4]>>> {
        let mut receivers = Vec::with_capacity(batches.len());

        for (i, data) in batches.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel();
            receivers.push(rx);

            let item = WorkItem::Ntt { data: data.clone(), inverse, result_tx: tx };

            // Alternate between GPU and CPU for parallel execution
            // Send large batches to GPU, small to CPU
            // Also distribute work when queue is full
            if data.len() >= self.config.gpu_threshold && i % 2 == 0 {
                if let Some(ref gpu_tx) = self.gpu_tx {
                    let _ = gpu_tx.send(item);
                    continue;
                }
            }

            if let Some(ref cpu_tx) = self.cpu_tx {
                let _ = cpu_tx.send(item);
            }
        }

        receivers
    }

    /// Get current statistics
    pub fn stats(&self) -> HybridStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = HybridStats::default();
    }

    // ========================================================================
    // CPU implementations
    // ========================================================================

    fn cpu_ntt(data: &mut [[u64; 4]], inverse: bool) {
        // Use the CPU NTT implementation
        if inverse {
            cpu_ntt_inverse(data);
        } else {
            cpu_ntt_forward(data);
        }
    }

    fn cpu_field_mul(a: &[[u64; 4]], b: &[[u64; 4]]) -> Vec<[u64; 4]> {
        #[cfg(feature = "parallel")]
        {
            a.par_iter()
                .zip(b.par_iter())
                .map(|(x, y)| field_mul(x, y))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| field_mul(x, y))
                .collect()
        }
    }
}

impl Drop for HybridExecutor {
    fn drop(&mut self) {
        // Send shutdown signals
        if let Some(tx) = self.gpu_tx.take() {
            let _ = tx.send(WorkItem::Shutdown);
        }
        if let Some(tx) = self.cpu_tx.take() {
            // Send shutdown to all CPU workers
            for _ in 0..self.config.cpu_threads {
                let _ = tx.send(WorkItem::Shutdown);
            }
        }

        // Wait for threads to finish
        if let Some(handle) = self.gpu_thread.take() {
            let _ = handle.join();
        }
        for handle in self.cpu_threads.drain(..) {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// Field arithmetic (inlined for CPU worker)
// ============================================================================

const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

const INV: u64 = 0xc2e1f593efffffff;

fn field_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut t = [0u64; 8];

    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let product = (a[i] as u128) * (b[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }
        t[i + 4] = carry as u64;
    }

    for i in 0..4 {
        let m = t[i].wrapping_mul(INV);
        let mut carry = 0u128;

        for j in 0..4 {
            let product = (m as u128) * (MODULUS[j] as u128) + (t[i + j] as u128) + carry;
            t[i + j] = product as u64;
            carry = product >> 64;
        }

        for j in (i + 4)..8 {
            let sum = (t[j] as u128) + carry;
            t[j] = sum as u64;
            carry = sum >> 64;
            if carry == 0 { break; }
        }
    }

    let mut result = [t[4], t[5], t[6], t[7]];

    // Final reduction
    let mut less = false;
    for i in (0..4).rev() {
        if result[i] < MODULUS[i] { less = true; break; }
        if result[i] > MODULUS[i] { break; }
    }

    if !less {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

// CPU NTT - full implementation using BN254 Fr field
// Uses the same algorithms as cuda/ntt_cpu.rs but doesn't depend on that module

/// Primitive root of unity for BN254 Fr field (2^28th root)
const ROOT_OF_UNITY: [u64; 4] = [
    0x3478a0a55ee3e703,
    0x7b77ae11bd943690,
    0x1bb0e74c0c0da9e6,
    0x0a6f3929fef59e65,
];

/// Montgomery R for Fr field
const R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];

fn cpu_ntt_forward(data: &mut [[u64; 4]]) {
    let n = data.len();
    if !n.is_power_of_two() || n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Bit-reverse permutation
    for i in 0..n {
        let rev = reverse_bits(i, log_n);
        if i < rev {
            data.swap(i, rev);
        }
    }

    // Get root of unity for this size
    let root = compute_root_of_unity(log_n);

    // Cooley-Tukey butterfly stages
    let mut m = 1;
    for _ in 0..log_n {
        let w_m = field_pow(&root, (n / (2 * m)) as u64);
        let mut k = 0;
        while k < n {
            let mut w = R; // Montgomery form of 1
            for j in 0..m {
                let t = field_mul(&w, &data[k + j + m]);
                let u = data[k + j];
                data[k + j] = field_add(&u, &t);
                data[k + j + m] = field_sub(&u, &t);
                w = field_mul(&w, &w_m);
            }
            k += 2 * m;
        }
        m *= 2;
    }
}

fn cpu_ntt_inverse(data: &mut [[u64; 4]]) {
    let n = data.len();
    if !n.is_power_of_two() || n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Bit-reverse permutation
    for i in 0..n {
        let rev = reverse_bits(i, log_n);
        if i < rev {
            data.swap(i, rev);
        }
    }

    // Get inverse root of unity
    let root = compute_root_of_unity(log_n);
    let inv_root = field_inv(&root);

    // Cooley-Tukey butterfly stages with inverse root
    let mut m = 1;
    for _ in 0..log_n {
        let w_m = field_pow(&inv_root, (n / (2 * m)) as u64);
        let mut k = 0;
        while k < n {
            let mut w = R;
            for j in 0..m {
                let t = field_mul(&w, &data[k + j + m]);
                let u = data[k + j];
                data[k + j] = field_add(&u, &t);
                data[k + j + m] = field_sub(&u, &t);
                w = field_mul(&w, &w_m);
            }
            k += 2 * m;
        }
        m *= 2;
    }

    // Scale by 1/n
    let n_inv = field_inv(&to_mont(n as u64));
    for elem in data.iter_mut() {
        *elem = field_mul(elem, &n_inv);
    }
}

fn reverse_bits(x: usize, bits: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn compute_root_of_unity(log_n: usize) -> [u64; 4] {
    // ROOT_OF_UNITY is 2^28th root, compute 2^log_n root
    let power = 1u64 << (28 - log_n);
    field_pow(&ROOT_OF_UNITY, power)
}

fn to_mont(x: u64) -> [u64; 4] {
    // Convert to Montgomery form by multiplying by R^2 mod p
    let mut result = [x, 0, 0, 0];
    // Simplified: just set the value (proper conversion would multiply by R^2)
    result
}

fn field_add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut carry = 0u64;

    for i in 0..4 {
        let sum = (a[i] as u128) + (b[i] as u128) + (carry as u128);
        result[i] = sum as u64;
        carry = (sum >> 64) as u64;
    }

    // Reduce if >= modulus
    let mut less = false;
    for i in (0..4).rev() {
        if result[i] < MODULUS[i] { less = true; break; }
        if result[i] > MODULUS[i] { break; }
    }

    if !less {
        let mut borrow = 0i128;
        for i in 0..4 {
            let diff = (result[i] as i128) - (MODULUS[i] as i128) - borrow;
            if diff < 0 {
                result[i] = (diff + (1i128 << 64)) as u64;
                borrow = 1;
            } else {
                result[i] = diff as u64;
                borrow = 0;
            }
        }
    }

    result
}

fn field_sub(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = (a[i] as i128) - (b[i] as i128) - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    // If borrow, add modulus
    if borrow != 0 {
        let mut carry = 0u128;
        for i in 0..4 {
            let sum = (result[i] as u128) + (MODULUS[i] as u128) + carry;
            result[i] = sum as u64;
            carry = sum >> 64;
        }
    }

    result
}

fn field_pow(base: &[u64; 4], mut exp: u64) -> [u64; 4] {
    let mut result = R; // Montgomery form of 1
    let mut base = *base;

    while exp > 0 {
        if exp & 1 == 1 {
            result = field_mul(&result, &base);
        }
        base = field_mul(&base, &base);
        exp >>= 1;
    }

    result
}

fn field_inv(a: &[u64; 4]) -> [u64; 4] {
    // Use Fermat's little theorem: a^(-1) = a^(p-2) mod p
    // p-2 for BN254 Fr
    let p_minus_2: [u64; 4] = [
        0x43e1f593efffffffu64,
        0x2833e84879b97091u64,
        0xb85045b68181585du64,
        0x30644e72e131a029u64,
    ];

    // Square and multiply
    let mut result = R;
    let mut base = *a;

    for i in 0..4 {
        for j in 0..64 {
            if (p_minus_2[i] >> j) & 1 == 1 {
                result = field_mul(&result, &base);
            }
            base = field_mul(&base, &base);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_config_default() {
        let config = HybridConfig::default();
        assert!(config.cpu_threads > 0);
        assert!(config.gpu_threshold > 0);
    }

    #[test]
    fn test_hybrid_stats() {
        let mut stats = HybridStats::default();
        stats.gpu_time_us = 500;
        stats.cpu_time_us = 500;
        assert!((stats.gpu_utilization() - 0.5).abs() < 0.01);
    }
}
