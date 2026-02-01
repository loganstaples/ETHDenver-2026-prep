//! Hybrid CPU+GPU Execution Demo
//!
//! Demonstrates how to maximize hardware utilization by running CPU and GPU in parallel.
//!
//! Run with: cargo run --example hybrid_execution --features metal --release
//!
//! This example shows:
//! 1. GPU handles large batch operations (NTT, field arithmetic)
//! 2. CPU handles smaller tasks in parallel
//! 3. Both execute simultaneously for maximum throughput

use std::time::{Duration, Instant};
use std::sync::mpsc;
use std::thread;

fn main() {
    println!("=== Hybrid CPU+GPU Execution Demo ===\n");
    println!("This demonstrates maximal hardware utilization by running");
    println!("CPU and GPU in parallel with intelligent work distribution.\n");

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("Note: Running in CPU-only mode (no Metal support)");
        run_cpu_only_demo();
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    run_hybrid_demo();
}

// ============================================================================
// CPU NTT Implementation (embedded for portability)
// ============================================================================

#[allow(dead_code)]
const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

#[allow(dead_code)]
const INV: u64 = 0xc2e1f593efffffff;

#[allow(dead_code)]
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

fn cpu_ntt_forward(data: &mut [[u64; 4]]) {
    let n = data.len();
    if !n.is_power_of_two() || n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Bit-reverse permutation
    for i in 0..n {
        let j = reverse_bits(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly stages (simplified - real impl would use proper roots of unity)
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        for start in (0..n).step_by(len) {
            for j in 0..half {
                let u = data[start + j];
                let v = data[start + j + half];
                // Simplified butterfly (real impl needs twiddle factors)
                data[start + j] = field_add(&u, &v);
                data[start + j + half] = field_sub(&u, &v);
            }
        }
        len *= 2;
    }
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

fn reverse_bits(x: usize, bits: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ============================================================================
// Demo Functions
// ============================================================================

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn run_cpu_only_demo() {
    use helix_prover::gpu::hybrid_executor::{HybridExecutor, HybridConfig};

    let config = HybridConfig {
        gpu_threshold: 1024,
        cpu_threads: num_cpus::get(),
        gpu_queue_depth: 3,
        work_stealing: true,
    };

    println!("Configuration:");
    println!("  CPU threads: {}", config.cpu_threads);
    println!("  GPU threshold: {} elements\n", config.gpu_threshold);

    let executor = HybridExecutor::new(config);

    let small_batch_count = 100;
    let large_batch_count = 10;

    println!("Submitting {} small batches (512 elements each)...", small_batch_count);
    println!("Submitting {} large batches (4096 elements each)...", large_batch_count);

    let start = Instant::now();

    let small_receivers: Vec<_> = (0..small_batch_count)
        .map(|i| {
            let data: Vec<[u64; 4]> = (0..512)
                .map(|j| [(i * 512 + j) as u64, 0, 0, 0])
                .collect();
            executor.submit_ntt(data, false)
        })
        .collect();

    let large_receivers: Vec<_> = (0..large_batch_count)
        .map(|i| {
            let data: Vec<[u64; 4]> = (0..4096)
                .map(|j| [(i * 4096 + j) as u64, 0, 0, 0])
                .collect();
            executor.submit_ntt(data, false)
        })
        .collect();

    let mut completed = 0;
    for rx in small_receivers {
        let _ = rx.recv();
        completed += 1;
    }
    for rx in large_receivers {
        let _ = rx.recv();
        completed += 1;
    }

    let elapsed = start.elapsed();
    let stats = executor.stats();

    println!("\nResults:");
    println!("  Completed: {} operations", completed);
    println!("  Total time: {:?}", elapsed);
    println!("  CPU operations: {}", stats.cpu_operations);
    println!("  GPU operations: {}", stats.gpu_operations);
    println!("  Elements processed: {}", stats.cpu_elements + stats.gpu_elements);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn run_hybrid_demo() {
    use std::sync::Arc;
    use helix_prover::metal::{MetalDevice, ntt::{MetalNtt, NttConfig}};

    println!("--- Phase 1: Sequential Baseline ---");
    println!("Running GPU and CPU operations sequentially...\n");

    // Initialize Metal
    let device = match MetalDevice::new() {
        Ok(d) => Arc::new(d),
        Err(e) => {
            println!("Failed to init Metal: {:?}", e);
            return;
        }
    };

    let mut ntt_config = NttConfig::default();
    ntt_config.min_gpu_batch_size = 64;

    let mut gpu_ntt = match MetalNtt::with_config(device.clone(), ntt_config) {
        Ok(n) => n,
        Err(e) => {
            println!("Failed to create NTT: {:?}", e);
            return;
        }
    };

    // Test parameters
    let gpu_batch_size = 1 << 16; // 64K elements for GPU
    let cpu_batch_size = 1 << 12; // 4K elements per CPU task
    let cpu_task_count = 64; // Number of CPU tasks
    let iterations = 5;

    // Create test data
    let gpu_data: Vec<[u64; 4]> = (0..gpu_batch_size)
        .map(|i| [(i as u64).wrapping_mul(0x12345), 0, 0, 0])
        .collect();

    let cpu_data: Vec<Vec<[u64; 4]>> = (0..cpu_task_count)
        .map(|batch| {
            (0..cpu_batch_size)
                .map(|i| [((batch * cpu_batch_size + i) as u64).wrapping_mul(0x67890), 0, 0, 0])
                .collect()
        })
        .collect();

    // Sequential execution: GPU first, then CPU
    let seq_start = Instant::now();

    for _ in 0..iterations {
        // GPU NTT
        let mut data = gpu_data.clone();
        let _ = gpu_ntt.forward(&mut data);

        // CPU NTTs
        for cpu_batch in &cpu_data {
            let mut data = cpu_batch.clone();
            cpu_ntt_forward(&mut data);
        }
    }

    let seq_elapsed = seq_start.elapsed();
    let total_elements = (gpu_batch_size + cpu_batch_size * cpu_task_count) * iterations;

    println!("Sequential Results:");
    println!("  Total time: {:?}", seq_elapsed);
    println!("  Elements processed: {}", total_elements);
    println!("  Throughput: {:.2} M elements/sec",
        total_elements as f64 / seq_elapsed.as_secs_f64() / 1_000_000.0);

    // Parallel execution: GPU and CPU simultaneously
    println!("\n--- Phase 2: Parallel Execution ---");
    println!("Running GPU and CPU operations simultaneously...\n");

    let par_start = Instant::now();

    for iter in 0..iterations {
        // Spawn CPU worker threads
        let cpu_handles: Vec<_> = cpu_data.iter().cloned().map(|mut data| {
            thread::spawn(move || {
                cpu_ntt_forward(&mut data);
                data
            })
        }).collect();

        // Run GPU NTT on main thread
        let mut gpu_data_copy = gpu_data.clone();
        let _ = gpu_ntt.forward(&mut gpu_data_copy);

        // Wait for CPU threads
        for handle in cpu_handles {
            let _ = handle.join();
        }

        if iter == 0 {
            println!("  First iteration complete");
        }
    }

    let par_elapsed = par_start.elapsed();

    println!("\nParallel Results:");
    println!("  Total time: {:?}", par_elapsed);
    println!("  Elements processed: {}", total_elements);
    println!("  Throughput: {:.2} M elements/sec",
        total_elements as f64 / par_elapsed.as_secs_f64() / 1_000_000.0);

    let speedup = seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64();
    println!("\n  Speedup from parallelization: {:.2}x", speedup);

    // Phase 3: Sustained parallel load
    println!("\n--- Phase 3: Sustained Parallel Load (10 seconds) ---");
    println!("Watch Activity Monitor - both CPU and GPU should be active!\n");

    let duration = Duration::from_secs(10);
    let start = Instant::now();

    // Stats tracking
    let gpu_ops = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let cpu_ops = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Spawn CPU worker pool
    let cpu_threads = num_cpus::get().saturating_sub(2).max(1);
    let (cpu_tx, cpu_rx) = mpsc::channel::<Option<Vec<[u64; 4]>>>();
    let cpu_rx = Arc::new(std::sync::Mutex::new(cpu_rx));

    let cpu_handles: Vec<_> = (0..cpu_threads).map(|_| {
        let rx = cpu_rx.clone();
        let ops = cpu_ops.clone();
        thread::spawn(move || {
            loop {
                let item = {
                    let rx = rx.lock().unwrap();
                    rx.recv().ok()
                };

                match item {
                    Some(Some(mut data)) => {
                        cpu_ntt_forward(&mut data);
                        ops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    _ => break,
                }
            }
        })
    }).collect();

    // Feed CPU workers in background
    let cpu_tx_clone = cpu_tx.clone();
    let start_clone = start;
    let cpu_feeder = thread::spawn(move || {
        let small_batch: Vec<[u64; 4]> = (0..cpu_batch_size)
            .map(|i| [(i as u64).wrapping_mul(0x67890), 0, 0, 0])
            .collect();

        while start_clone.elapsed() < duration {
            for _ in 0..cpu_threads {
                if cpu_tx_clone.send(Some(small_batch.clone())).is_err() {
                    return;
                }
            }
        }
    });

    // GPU loop on main thread
    let mut gpu_data_sustained: Vec<[u64; 4]> = (0..gpu_batch_size)
        .map(|i| [(i as u64).wrapping_mul(0x12345), 0, 0, 0])
        .collect();

    let mut dot_count = 0;
    while start.elapsed() < duration {
        // Reset data
        for (i, elem) in gpu_data_sustained.iter_mut().enumerate() {
            elem[0] = (i as u64).wrapping_mul(0x12345).wrapping_add(
                gpu_ops.load(std::sync::atomic::Ordering::Relaxed)
            );
        }

        let _ = gpu_ntt.forward(&mut gpu_data_sustained);
        gpu_ops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        dot_count += 1;
        if dot_count % 20 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    // Shutdown
    drop(cpu_tx);
    let _ = cpu_feeder.join();
    for handle in cpu_handles {
        let _ = handle.join();
    }

    let final_gpu_ops = gpu_ops.load(std::sync::atomic::Ordering::Relaxed);
    let final_cpu_ops = cpu_ops.load(std::sync::atomic::Ordering::Relaxed);

    println!("\n\nSustained Load Results:");
    println!("  Duration: 10 seconds");
    println!("  GPU operations: {} ({:.1}/sec)", final_gpu_ops, final_gpu_ops as f64 / 10.0);
    println!("  CPU operations: {} ({:.1}/sec)", final_cpu_ops, final_cpu_ops as f64 / 10.0);
    println!("  Total operations: {}", final_gpu_ops + final_cpu_ops);

    let gpu_elements = final_gpu_ops * gpu_batch_size as u64;
    let cpu_elements = final_cpu_ops * cpu_batch_size as u64;
    let total_elements = gpu_elements + cpu_elements;

    println!("\n  GPU elements: {} M", gpu_elements / 1_000_000);
    println!("  CPU elements: {} M", cpu_elements / 1_000_000);
    println!("  Total throughput: {:.2} M elements/sec", total_elements as f64 / 10.0 / 1_000_000.0);

    // Final stats
    let ntt_stats = gpu_ntt.stats();
    println!("\nGPU Stats:");
    println!("  Forward transforms: {}", ntt_stats.forward_transforms);
    println!("  GPU time: {} ms", ntt_stats.gpu_time_us / 1000);

    println!("\n=== Demo Complete ===");
    println!("\nKey takeaways:");
    println!("1. Large batches (64K+) go to GPU for maximum throughput");
    println!("2. Small batches run on CPU threads in parallel");
    println!("3. CPU prepares data while GPU processes");
    println!("4. Both utilize their respective hardware simultaneously");
}
