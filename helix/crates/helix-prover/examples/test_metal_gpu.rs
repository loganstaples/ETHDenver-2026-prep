//! Test that Metal GPU acceleration is actually being used.
//!
//! Run with: cargo run --example test_metal_gpu --features metal

use std::time::Instant;

fn main() {
    println!("=== Metal GPU Acceleration Test ===\n");

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("ERROR: This test requires macOS with the 'metal' feature enabled.");
        println!("Run with: cargo run --example test_metal_gpu --features metal");
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        test_metal_ntt();
        test_metal_field_ops();
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_metal_ntt() {
    use std::sync::Arc;
    use helix_prover::metal::{MetalDevice, ntt::{MetalNtt, NttConfig}};

    println!("--- Testing Metal NTT ---");

    // Create Metal device
    let device = match MetalDevice::new() {
        Ok(d) => Arc::new(d),
        Err(e) => {
            println!("Failed to create MetalDevice: {:?}", e);
            return;
        }
    };

    println!("Metal device created successfully");

    // Create NTT engine with low threshold to force GPU usage
    let mut config = NttConfig::default();
    config.min_gpu_batch_size = 64; // Force GPU for smaller sizes

    let mut ntt = match MetalNtt::with_config(device, config) {
        Ok(n) => n,
        Err(e) => {
            println!("Failed to create MetalNtt: {:?}", e);
            return;
        }
    };

    println!("MetalNtt created successfully\n");

    // Test with various sizes
    for log_n in [10, 12, 14, 16] {
        let n = 1 << log_n;

        // Create random data (in Montgomery form)
        let mut data: Vec<[u64; 4]> = (0..n)
            .map(|i| [(i as u64).wrapping_mul(0x12345), 0, 0, 0])
            .collect();

        let start = Instant::now();
        match ntt.forward(&mut data) {
            Ok(()) => {
                let elapsed = start.elapsed();
                println!("  NTT size 2^{} ({} elements): {:?}", log_n, n, elapsed);
            }
            Err(e) => {
                println!("  NTT size 2^{} FAILED: {:?}", log_n, e);
            }
        }
    }

    // Print stats
    let stats = ntt.stats();
    println!("\nNTT Stats:");
    println!("  Forward transforms: {}", stats.forward_transforms);
    println!("  GPU time: {} us", stats.gpu_time_us);
    println!("  Elements processed: {}", stats.elements_processed);

    println!();
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn test_metal_field_ops() {
    use helix_prover::metal::field_ops::{MetalFieldOps, BatchFieldOperation};
    use helix_prover::gkr::FieldElement;

    println!("--- Testing Metal Field Operations ---");

    // Create field ops with low threshold
    let mut field_ops = MetalFieldOps::with_min_batch_size(64);

    println!("MetalFieldOps created successfully");
    println!("GPU available: {}", field_ops.is_gpu_available());

    // Create test data
    let batch_sizes = [1000, 10000, 100000];

    for batch_size in batch_sizes {
        let a: Vec<FieldElement> = (0..batch_size)
            .map(|i| FieldElement::from_raw([(i as u64).wrapping_mul(0x12345), 0, 0, 0]))
            .collect();
        let b: Vec<FieldElement> = (0..batch_size)
            .map(|i| FieldElement::from_raw([(i as u64).wrapping_mul(0x67890), 0, 0, 0]))
            .collect();

        let op = BatchFieldOperation::mul(a, b);

        let start = Instant::now();
        match field_ops.execute(&op) {
            Ok(_result) => {
                let elapsed = start.elapsed();
                println!("  Batch multiply {} elements: {:?} ({:.2} M ops/sec)",
                    batch_size,
                    elapsed,
                    (batch_size as f64) / elapsed.as_secs_f64() / 1_000_000.0
                );
            }
            Err(e) => {
                println!("  Batch multiply {} FAILED: {:?}", batch_size, e);
            }
        }
    }

    // Print stats
    let stats = field_ops.stats();
    println!("\nField Ops Stats:");
    println!("  GPU dispatches: {}", stats.num_dispatches);
    println!("  GPU time: {} us", stats.gpu_time_us);
    println!("  Bytes transferred: {}", stats.bytes_transferred);
}
