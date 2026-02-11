//! Stress and concurrency tests for helix-mpc.
//!
//! Validates that the system handles:
//! - Concurrent pool access from multiple threads
//! - Large-scale triple generation
//! - Pipeline under load

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use helix_mpc::beaver::dealer::TrustedDealer;
use helix_mpc::beaver::pool::BeaverPool;
use helix_mpc::beaver::pipeline::{BeaverPipeline, PipelineConfig, TripleSource};
use helix_mpc::field::Fr;

// =============================================================================
// Concurrent Pool Access
// =============================================================================

#[test]
fn test_concurrent_pool_take_scalar() {
    // 10 threads each consuming from a shared pool — no panics allowed.
    let mut dealer = TrustedDealer::with_seed(42);
    let per_party = dealer.generate_scalar_triples(1000, 3);

    let pool = Arc::new(Mutex::new({
        let mut p = BeaverPool::new(0, 3, 64);
        p.fill_scalar(per_party[0].clone());
        p
    }));

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let pool = pool.clone();
            thread::spawn(move || {
                let mut successes = 0;
                for _ in 0..100 {
                    let mut p = pool.lock().unwrap();
                    if p.take_scalar().is_ok() {
                        successes += 1;
                    }
                }
                successes
            })
        })
        .collect();

    let total_consumed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // All 1000 triples should be consumed across threads.
    assert_eq!(total_consumed, 1000);

    // Pool should be empty.
    assert_eq!(pool.lock().unwrap().scalar_available(), 0);
}

#[test]
fn test_concurrent_pool_no_panic() {
    // Mix of take and fill operations from multiple threads.
    let pool = Arc::new(Mutex::new(BeaverPool::new(0, 3, 64)));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let pool = pool.clone();
            thread::spawn(move || {
                let mut dealer = TrustedDealer::with_seed(i as u64);
                for _ in 0..50 {
                    if i % 2 == 0 {
                        // Producer: generate and fill.
                        let triples = dealer.generate_scalar_triples(10, 3);
                        pool.lock().unwrap().fill_scalar(triples[0].clone());
                    } else {
                        // Consumer: take.
                        let _ = pool.lock().unwrap().take_scalar();
                    }
                }
            })
        })
        .collect();

    // No panics.
    for h in handles {
        h.join().unwrap();
    }
}

// =============================================================================
// Large-Scale Triple Generation
// =============================================================================

#[test]
fn test_100k_triples_correctness() {
    let mut dealer = TrustedDealer::with_seed(42);
    let num_parties = 3;
    let count = 100_000;

    let batch = dealer.generate_scalar_triples(count, num_parties);

    // Verify all triples: sum(a) * sum(b) == sum(c).
    // Sample every 1000th triple for speed (100 checks).
    for idx in (0..count).step_by(1000) {
        let a: Fr = (0..num_parties)
            .map(|p| batch[p][idx].a.clone())
            .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
        let b: Fr = (0..num_parties)
            .map(|p| batch[p][idx].b.clone())
            .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
        let c: Fr = (0..num_parties)
            .map(|p| batch[p][idx].c.clone())
            .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));

        // TrustedDealer uses mpc_scale (field-exact fixed-point scaling).
        let ab = a.mpc_scale(&b);
        assert!(
            ab.ct_eq(&c).to_bool(),
            "Triple {} failed: a*b != c",
            idx,
        );
    }
}

// =============================================================================
// Pool Split/Merge Under Load
// =============================================================================

#[test]
fn test_pool_split_concurrent_consumption() {
    // Simulate parallel attention: split pool, consume from sub-pools
    // in parallel, then merge back.
    let mut dealer = TrustedDealer::with_seed(42);
    let per_party = dealer.generate_scalar_triples(400, 3);

    let mut pool = BeaverPool::new(0, 3, 64);
    pool.fill_scalar(per_party[0].clone());

    let num_heads = 4;
    let sub_pools: Vec<_> = pool
        .split(num_heads)
        .into_iter()
        .map(|p| Arc::new(Mutex::new(p)))
        .collect();

    // Each head consumes 50 triples in its own thread.
    let handles: Vec<_> = sub_pools
        .iter()
        .map(|sub| {
            let sub = sub.clone();
            thread::spawn(move || {
                let mut consumed = 0;
                for _ in 0..50 {
                    if sub.lock().unwrap().take_scalar().is_ok() {
                        consumed += 1;
                    }
                }
                consumed
            })
        })
        .collect();

    let total_consumed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total_consumed, 200); // 4 heads * 50 each

    // Merge back.
    let recovered: Vec<BeaverPool> = sub_pools
        .into_iter()
        .map(|arc| Arc::try_unwrap(arc).unwrap().into_inner().unwrap())
        .collect();

    pool.merge(recovered);

    // Should have 200 remaining (400 - 200 consumed).
    assert_eq!(pool.scalar_available(), 200);
}

// =============================================================================
// Pipeline Under Load
// =============================================================================

#[test]
fn test_pipeline_ensure_available() {
    let config = PipelineConfig::small_model();
    let pipeline = BeaverPipeline::new(3, config);
    pipeline.start_replenishment();

    // Wait for initial generation to start.
    thread::sleep(Duration::from_millis(100));

    // Request 100 triples with timeout.
    let result = pipeline.ensure_available(100, Duration::from_secs(5));
    assert!(
        result.is_ok(),
        "ensure_available should succeed within timeout: {:?}",
        result,
    );

    pipeline.shutdown();
}

#[test]
fn test_pipeline_distributed_mode() {
    let config = PipelineConfig::distributed(2);
    let pipeline = BeaverPipeline::new(3, config);
    pipeline.start_replenishment();

    thread::sleep(Duration::from_millis(100));

    let result = pipeline.ensure_available(50, Duration::from_secs(5));
    assert!(
        result.is_ok(),
        "Distributed pipeline should produce triples: {:?}",
        result,
    );

    pipeline.shutdown();
}

// =============================================================================
// Matrix Triple Generation
// =============================================================================

#[test]
fn test_large_matrix_triple_generation() {
    let mut dealer = TrustedDealer::with_seed(42);
    let num_parties = 3;

    // Generate a [64, 64, 64] matrix triple.
    let triples = dealer.generate_matrix_triple(64, 64, 64, num_parties);
    assert_eq!(triples.len(), num_parties);

    for party_triple in &triples {
        assert_eq!(party_triple.m, 64);
        assert_eq!(party_triple.k, 64);
        assert_eq!(party_triple.n, 64);
    }

    // Also test rectangular: [128, 64, 32].
    let rect_triples = dealer.generate_matrix_triple(128, 64, 32, num_parties);
    assert_eq!(rect_triples.len(), num_parties);
    assert_eq!(rect_triples[0].m, 128);
    assert_eq!(rect_triples[0].k, 64);
    assert_eq!(rect_triples[0].n, 32);
}
