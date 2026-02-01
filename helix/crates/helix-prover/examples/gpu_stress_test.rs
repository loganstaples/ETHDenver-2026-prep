//! GPU Stress Test - Verify Metal GPU is actually being utilized
//!
//! Run with: cargo run --example gpu_stress_test --features metal --release

use std::time::{Duration, Instant};

fn main() {
    println!("=== GPU Stress Test ===\n");

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("ERROR: Requires macOS with 'metal' feature");
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    run_stress_test();
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn run_stress_test() {
    use std::sync::Arc;
    use helix_prover::metal::{MetalDevice, ntt::{MetalNtt, NttConfig}};
    use helix_prover::metal::field_ops::{MetalFieldOps, BatchFieldOperation};
    use helix_prover::gkr::FieldElement;

    // Create Metal device
    let device = match MetalDevice::new() {
        Ok(d) => Arc::new(d),
        Err(e) => {
            println!("Failed to create MetalDevice: {:?}", e);
            return;
        }
    };

    println!("Metal device initialized\n");

    // =========================================================================
    // Test 1: Large NTT operations
    // =========================================================================
    println!("--- Test 1: Large NTT Operations ---");
    println!("Running 20 iterations of 2^18 (262K element) NTT...\n");

    let mut config = NttConfig::default();
    config.min_gpu_batch_size = 64;

    let mut ntt = match MetalNtt::with_config(device.clone(), config) {
        Ok(n) => n,
        Err(e) => {
            println!("Failed to create MetalNtt: {:?}", e);
            return;
        }
    };

    let n = 1 << 18; // 262K elements
    let iterations = 20;

    // Create large data
    let base_data: Vec<[u64; 4]> = (0..n)
        .map(|i| [(i as u64).wrapping_mul(0x12345), 0, 0, 0])
        .collect();

    let start = Instant::now();
    for i in 0..iterations {
        let mut data = base_data.clone();
        match ntt.forward(&mut data) {
            Ok(()) => {
                if i == 0 {
                    println!("  First iteration complete");
                }
            }
            Err(e) => {
                println!("  NTT failed at iteration {}: {:?}", i, e);
                break;
            }
        }
    }
    let elapsed = start.elapsed();

    println!("\n  Total time: {:?}", elapsed);
    println!("  Per iteration: {:?}", elapsed / iterations);
    println!("  Throughput: {:.2} M elements/sec",
        (n as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0);

    let stats = ntt.stats();
    println!("\n  GPU Stats:");
    println!("    Forward transforms: {}", stats.forward_transforms);
    println!("    GPU time: {} ms", stats.gpu_time_us / 1000);
    println!("    Elements processed: {}", stats.elements_processed);

    // =========================================================================
    // Test 2: Large batch field multiplications
    // =========================================================================
    println!("\n--- Test 2: Large Batch Field Multiplications ---");
    println!("Running 50 iterations of 1M element multiplications...\n");

    let mut field_ops = MetalFieldOps::with_min_batch_size(64);

    let batch_size = 1_000_000;
    let iterations = 50;

    let a: Vec<FieldElement> = (0..batch_size)
        .map(|i| FieldElement::from_raw([(i as u64).wrapping_mul(0x12345), 0, 0, 0]))
        .collect();
    let b: Vec<FieldElement> = (0..batch_size)
        .map(|i| FieldElement::from_raw([(i as u64).wrapping_mul(0x67890), 0, 0, 0]))
        .collect();

    let start = Instant::now();
    for i in 0..iterations {
        let op = BatchFieldOperation::mul(a.clone(), b.clone());
        match field_ops.execute(&op) {
            Ok(_) => {
                if i == 0 {
                    println!("  First iteration complete");
                }
            }
            Err(e) => {
                println!("  Field mul failed at iteration {}: {:?}", i, e);
                break;
            }
        }
    }
    let elapsed = start.elapsed();

    println!("\n  Total time: {:?}", elapsed);
    println!("  Per iteration: {:?}", elapsed / iterations);
    println!("  Throughput: {:.2} M ops/sec",
        (batch_size as f64 * iterations as f64) / elapsed.as_secs_f64() / 1_000_000.0);

    let stats = field_ops.stats();
    println!("\n  GPU Stats:");
    println!("    GPU dispatches: {}", stats.num_dispatches);
    println!("    GPU time: {} ms", stats.gpu_time_us / 1000);
    println!("    Bytes transferred: {} MB", stats.bytes_transferred / 1_000_000);

    // =========================================================================
    // Test 3: Sustained GPU load (10 seconds)
    // =========================================================================
    println!("\n--- Test 3: Sustained GPU Load (10 seconds) ---");
    println!("Watch Activity Monitor GPU usage now!\n");

    let duration = Duration::from_secs(10);
    let n = 1 << 16; // 64K elements
    let mut data: Vec<[u64; 4]> = (0..n)
        .map(|i| [(i as u64).wrapping_mul(0x12345), 0, 0, 0])
        .collect();

    let start = Instant::now();
    let mut count = 0;

    while start.elapsed() < duration {
        // Reset data each iteration
        for (i, elem) in data.iter_mut().enumerate() {
            elem[0] = (i as u64).wrapping_mul(0x12345).wrapping_add(count);
        }

        let _ = ntt.forward(&mut data);
        count += 1;

        if count % 100 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\n\n  Completed {} NTT operations in 10 seconds", count);
    println!("  Rate: {:.1} ops/sec", count as f64 / 10.0);

    let final_stats = ntt.stats();
    println!("\n  Final GPU Stats:");
    println!("    Total transforms: {}", final_stats.forward_transforms);
    println!("    Total GPU time: {} ms", final_stats.gpu_time_us / 1000);
}
