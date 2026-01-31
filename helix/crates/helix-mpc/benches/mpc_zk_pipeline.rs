//! Performance Benchmarks for MPC-ZK Integration Pipeline
//!
//! These benchmarks measure the performance of the complete HELIX MPC-ZK pipeline:
//! - Secret sharing of model weights
//! - Share validity proof generation and verification
//! - Gradient aggregation proofs
//! - MAC verification proofs
//! - Batch verification
//!
//! Run with: `cargo bench --package helix-mpc`
//!
//! Results help optimize:
//! - Proof generation throughput
//! - Verification latency
//! - Memory usage
//! - Scaling characteristics

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime,
    BenchmarkGroup, BenchmarkId, Criterion, Throughput,
};
use helix_mpc::{
    error::MPCResult,
    field::Fr,
    proofs::{
        BatchedProof, BatchVerifier, MPCProof,
        ShareValidityProof, ShareValidityProver, ShareValidityVerifier, ShareValidityWitness,
        AggregationProof, AggregationProver, AggregationVerifier, GradientAggregationWitness,
        GradientShareInput,
        MACProof, MACProver, ZKMACVerifier, MACWitness,
    },
    types::PartyId,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::time::Duration;

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Party counts to benchmark.
const PARTY_COUNTS: &[usize] = &[3, 5, 7, 10];

/// Model parameter counts to benchmark.
const PARAM_COUNTS: &[usize] = &[128, 512, 2048, 8192];

/// Gradient dimensions to benchmark.
const GRADIENT_DIMS: &[usize] = &[32, 128, 512, 2048];

/// Value counts for MAC verification.
const MAC_VALUE_COUNTS: &[usize] = &[16, 64, 256, 1024];

/// Batch sizes for aggregation.
const BATCH_SIZES: &[usize] = &[4, 16, 64];

// =============================================================================
// Helper Functions
// =============================================================================

/// Computes a hash commitment to field elements.
fn compute_commitment(data: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(&v.to_bytes_le());
    }
    hasher.update(blinding);
    hasher.finalize().into()
}

/// Creates random field elements.
fn random_field_elements(count: usize, rng: &mut impl Rng) -> Vec<Fr> {
    (0..count).map(|_| Fr::random(rng)).collect()
}

/// Creates a share validity witness.
fn create_share_validity_witness(
    num_elements: usize,
    rng: &mut impl Rng,
) -> ShareValidityWitness {
    let share_values = random_field_elements(num_elements, rng);
    let mut blinding = [0u8; 32];
    rng.fill(&mut blinding);
    let commitment = compute_commitment(&share_values, &blinding);
    let mut dealer_pk = [0u8; 32];
    rng.fill(&mut dealer_pk);

    ShareValidityWitness {
        share_values,
        shape: vec![num_elements],
        party: PartyId::new("party_0"),
        blinding,
        commitment,
        min_value: Fr::from_f64(-1e10),
        max_value: Fr::from_f64(1e10),
        dealer_public_key: dealer_pk,
        dealer_signature: vec![0u8; 64],
    }
}

/// Creates a gradient aggregation witness.
fn create_aggregation_witness(
    num_parties: usize,
    num_gradients: usize,
    rng: &mut impl Rng,
) -> GradientAggregationWitness {
    let mut witness = GradientAggregationWitness::new(num_parties, num_gradients, 0);

    for party_idx in 0..num_parties {
        let values = random_field_elements(num_gradients, rng);
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&values, &blinding);

        let input = GradientShareInput {
            party: PartyId::new(format!("party_{}", party_idx)),
            values,
            commitment,
            blinding,
        };
        witness.add_gradient_share(input).unwrap();
    }

    witness.compute_aggregation();
    witness
}

/// Creates a MAC witness with properly constructed SPDZ-style MACs.
/// For SPDZ, the MAC relationship is: Σ m_i = α · Σ x_i
fn create_mac_witness(
    num_parties: usize,
    num_values: usize,
    rng: &mut impl Rng,
) -> MACWitness {
    let mut witness = MACWitness::new(num_parties, num_values);

    // Generate alpha shares that sum to global_alpha.
    let global_alpha = Fr::random(rng);
    let mut alpha_sum = Fr::ZERO;
    let mut alpha_shares = Vec::with_capacity(num_parties);
    for _ in 0..num_parties - 1 {
        let share = Fr::random(rng);
        alpha_sum = Fr::add(&alpha_sum, &share);
        alpha_shares.push(share);
    }
    alpha_shares.push(Fr::sub(&global_alpha, &alpha_sum));
    witness.set_alpha_shares(alpha_shares).unwrap();

    // Generate actual values and compute expected MACs.
    let actual_values: Vec<Fr> = random_field_elements(num_values, rng);
    let expected_macs: Vec<Fr> = actual_values.iter()
        .map(|v| Fr::mul(&global_alpha, v))
        .collect();

    // Create additive shares for values.
    let mut all_value_shares: Vec<Vec<Fr>> = Vec::with_capacity(num_parties);
    let mut value_sums = vec![Fr::ZERO; num_values];
    for _ in 0..num_parties - 1 {
        let shares = random_field_elements(num_values, rng);
        for (i, s) in shares.iter().enumerate() {
            value_sums[i] = Fr::add(&value_sums[i], s);
        }
        all_value_shares.push(shares);
    }
    all_value_shares.push(
        (0..num_values).map(|i| Fr::sub(&actual_values[i], &value_sums[i])).collect()
    );

    // Create additive shares for MACs.
    let mut all_mac_shares: Vec<Vec<Fr>> = Vec::with_capacity(num_parties);
    let mut mac_sums = vec![Fr::ZERO; num_values];
    for _ in 0..num_parties - 1 {
        let shares = random_field_elements(num_values, rng);
        for (i, s) in shares.iter().enumerate() {
            mac_sums[i] = Fr::add(&mac_sums[i], s);
        }
        all_mac_shares.push(shares);
    }
    all_mac_shares.push(
        (0..num_values).map(|i| Fr::sub(&expected_macs[i], &mac_sums[i])).collect()
    );

    // Set party shares.
    for party_idx in 0..num_parties {
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        witness.set_party_shares(
            party_idx,
            all_value_shares[party_idx].clone(),
            all_mac_shares[party_idx].clone(),
            blinding,
        ).unwrap();
    }

    witness
}

// =============================================================================
// Share Validity Benchmarks
// =============================================================================

fn bench_share_validity_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("share_validity_proof_generation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    for &num_elements in PARAM_COUNTS {
        group.throughput(Throughput::Elements(num_elements as u64));
        group.bench_with_input(
            BenchmarkId::new("elements", num_elements),
            &num_elements,
            |b, &num_elements| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_share_validity_witness(num_elements, &mut rng);
                let mut prover = ShareValidityProver::with_seed(42);

                b.iter(|| {
                    black_box(prover.prove(&witness).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_share_validity_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("share_validity_proof_verification");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    for &num_elements in PARAM_COUNTS {
        group.throughput(Throughput::Elements(num_elements as u64));
        group.bench_with_input(
            BenchmarkId::new("elements", num_elements),
            &num_elements,
            |b, &num_elements| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_share_validity_witness(num_elements, &mut rng);
                let mut prover = ShareValidityProver::with_seed(42);
                let proof = prover.prove(&witness).unwrap();
                let verifier = ShareValidityVerifier::new();

                b.iter(|| {
                    black_box(verifier.verify(&proof).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_share_validity_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("share_validity_serialization");
    group.measurement_time(Duration::from_secs(5));

    for &num_elements in PARAM_COUNTS {
        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        let witness = create_share_validity_witness(num_elements, &mut rng);
        let mut prover = ShareValidityProver::with_seed(42);
        let proof = prover.prove(&witness).unwrap();

        group.bench_with_input(
            BenchmarkId::new("serialize", num_elements),
            &proof,
            |b, proof| {
                b.iter(|| black_box(proof.to_bytes()));
            },
        );

        let bytes = proof.to_bytes();
        group.bench_with_input(
            BenchmarkId::new("deserialize", num_elements),
            &bytes,
            |b, bytes| {
                b.iter(|| black_box(ShareValidityProof::from_bytes(bytes).unwrap()));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Aggregation Benchmarks
// =============================================================================

fn bench_aggregation_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation_proof_generation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &num_parties in PARTY_COUNTS {
        for &num_gradients in GRADIENT_DIMS {
            let id = format!("{}parties_{}grads", num_parties, num_gradients);
            group.throughput(Throughput::Elements((num_parties * num_gradients) as u64));

            group.bench_function(BenchmarkId::new("generate", &id), |b| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_aggregation_witness(num_parties, num_gradients, &mut rng);
                let mut prover = AggregationProver::with_seed(42);

                b.iter(|| {
                    black_box(prover.prove(&witness).unwrap())
                });
            });
        }
    }

    group.finish();
}

fn bench_aggregation_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation_proof_verification");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &num_parties in PARTY_COUNTS {
        for &num_gradients in &[128, 512] { // Subset for verification
            let id = format!("{}parties_{}grads", num_parties, num_gradients);

            group.bench_function(BenchmarkId::new("verify", &id), |b| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_aggregation_witness(num_parties, num_gradients, &mut rng);
                let mut prover = AggregationProver::with_seed(42);
                let proof = prover.prove(&witness).unwrap();
                let verifier = AggregationVerifier::new();

                b.iter(|| {
                    black_box(verifier.verify(&proof).unwrap())
                });
            });
        }
    }

    group.finish();
}

fn bench_aggregation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation_scaling");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);

    // Test scaling with party count.
    let num_gradients = 256;
    for &num_parties in PARTY_COUNTS {
        group.throughput(Throughput::Elements(num_parties as u64));
        group.bench_with_input(
            BenchmarkId::new("parties", num_parties),
            &num_parties,
            |b, &num_parties| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_aggregation_witness(num_parties, num_gradients, &mut rng);
                let mut prover = AggregationProver::with_seed(42);

                b.iter(|| {
                    black_box(prover.prove(&witness).unwrap())
                });
            },
        );
    }

    // Test scaling with gradient dimension.
    let num_parties = 5;
    for &num_gradients in GRADIENT_DIMS {
        group.throughput(Throughput::Elements(num_gradients as u64));
        group.bench_with_input(
            BenchmarkId::new("gradients", num_gradients),
            &num_gradients,
            |b, &num_gradients| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_aggregation_witness(num_parties, num_gradients, &mut rng);
                let mut prover = AggregationProver::with_seed(42);

                b.iter(|| {
                    black_box(prover.prove(&witness).unwrap())
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// MAC Verification Benchmarks
// =============================================================================

fn bench_mac_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mac_proof_generation");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &num_parties in &[3, 5, 7] {
        for &num_values in MAC_VALUE_COUNTS {
            let id = format!("{}parties_{}values", num_parties, num_values);
            group.throughput(Throughput::Elements((num_parties * num_values) as u64));

            group.bench_function(BenchmarkId::new("generate", &id), |b| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_mac_witness(num_parties, num_values, &mut rng);
                let mut prover = MACProver::with_seed(42);

                b.iter(|| {
                    black_box(prover.prove(&witness).unwrap())
                });
            });
        }
    }

    group.finish();
}

fn bench_mac_proof_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("mac_proof_verification");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &num_parties in &[3, 5, 7] {
        for &num_values in &[64, 256] { // Subset for verification
            let id = format!("{}parties_{}values", num_parties, num_values);

            group.bench_function(BenchmarkId::new("verify", &id), |b| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let witness = create_mac_witness(num_parties, num_values, &mut rng);
                let mut prover = MACProver::with_seed(42);
                let proof = prover.prove(&witness).unwrap();
                let verifier = ZKMACVerifier::new();

                b.iter(|| {
                    black_box(verifier.verify(&proof).unwrap())
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Batch Verification Benchmarks
// =============================================================================

fn bench_batch_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_verification");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);

    for &batch_size in BATCH_SIZES {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("proofs", batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);
                let mut batch = BatchedProof::new();

                // Add share validity proofs.
                let mut share_prover = ShareValidityProver::with_seed(42);
                for _ in 0..batch_size {
                    let witness = create_share_validity_witness(128, &mut rng);
                    let proof = share_prover.prove(&witness).unwrap();
                    batch.add_share_validity(proof);
                }

                batch.generate_challenges(42);
                batch.compute_aggregate();

                let verifier = BatchVerifier::new();

                b.iter(|| {
                    black_box(verifier.verify(&batch).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_mixed_proofs(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_mixed_proofs");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(30);

    // Create a batch with different proof types.
    let num_share_proofs = 3;
    let num_agg_proofs = 1;
    let num_mac_proofs = 1;
    let total_proofs = num_share_proofs + num_agg_proofs + num_mac_proofs;

    group.throughput(Throughput::Elements(total_proofs as u64));

    group.bench_function("create_and_verify", |b| {
        let mut rng = ChaCha20Rng::seed_from_u64(12345);

        b.iter(|| {
            let mut batch = BatchedProof::new();

            // Add share validity proofs.
            let mut share_prover = ShareValidityProver::with_seed(42);
            for _ in 0..num_share_proofs {
                let witness = create_share_validity_witness(128, &mut rng);
                let proof = share_prover.prove(&witness).unwrap();
                batch.add_share_validity(proof);
            }

            // Add aggregation proof.
            let agg_witness = create_aggregation_witness(3, 64, &mut rng);
            let mut agg_prover = AggregationProver::with_seed(42);
            let agg_proof = agg_prover.prove(&agg_witness).unwrap();
            batch.add_aggregation(agg_proof);

            // Add MAC proof.
            let mac_witness = create_mac_witness(3, 64, &mut rng);
            let mut mac_prover = MACProver::with_seed(42);
            let mac_proof = mac_prover.prove(&mac_witness).unwrap();
            batch.add_mac_verification(mac_proof);

            batch.generate_challenges(42);
            batch.compute_aggregate();

            let verifier = BatchVerifier::new();
            black_box(verifier.verify(&batch).unwrap())
        });
    });

    group.finish();
}

// =============================================================================
// End-to-End Pipeline Benchmarks
// =============================================================================

fn bench_full_training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_training_step");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);

    // Benchmark a complete training step with all proofs.
    for &num_parties in &[3, 5] {
        let num_params = 256;
        let id = format!("{}parties_{}params", num_parties, num_params);

        group.bench_function(BenchmarkId::new("complete", &id), |b| {
            let mut rng = ChaCha20Rng::seed_from_u64(12345);

            b.iter(|| {
                let mut batch = BatchedProof::new();

                // Share validity proofs for each party.
                for party_idx in 0..num_parties {
                    let witness = create_share_validity_witness(num_params, &mut rng);
                    let mut prover = ShareValidityProver::with_seed(42 + party_idx as u64);
                    let proof = prover.prove(&witness).unwrap();
                    batch.add_share_validity(proof);
                }

                // Aggregation proof.
                let agg_witness = create_aggregation_witness(num_parties, num_params, &mut rng);
                let mut agg_prover = AggregationProver::with_seed(42);
                let agg_proof = agg_prover.prove(&agg_witness).unwrap();
                batch.add_aggregation(agg_proof);

                // MAC proof.
                let mac_witness = create_mac_witness(num_parties, num_params, &mut rng);
                let mut mac_prover = MACProver::with_seed(42);
                let mac_proof = mac_prover.prove(&mac_witness).unwrap();
                batch.add_mac_verification(mac_proof);

                // Verify batch.
                batch.generate_challenges(42);
                batch.compute_aggregate();
                let verifier = BatchVerifier::new();
                black_box(verifier.verify(&batch).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_proof_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_size");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    // Measure proof sizes for different configurations.
    for &num_elements in PARAM_COUNTS {
        group.bench_function(BenchmarkId::new("share_validity", num_elements), |b| {
            let mut rng = ChaCha20Rng::seed_from_u64(12345);
            let witness = create_share_validity_witness(num_elements, &mut rng);
            let mut prover = ShareValidityProver::with_seed(42);

            b.iter(|| {
                let proof = prover.prove(&witness).unwrap();
                black_box(proof.size())
            });
        });
    }

    for &num_gradients in &[128, 512] {
        let id = format!("aggregation_{}", num_gradients);
        group.bench_function(BenchmarkId::new(&id, 5), |b| {
            let mut rng = ChaCha20Rng::seed_from_u64(12345);
            let witness = create_aggregation_witness(5, num_gradients, &mut rng);
            let mut prover = AggregationProver::with_seed(42);

            b.iter(|| {
                let proof = prover.prove(&witness).unwrap();
                black_box(proof.size())
            });
        });
    }

    group.finish();
}

// =============================================================================
// Memory Benchmarks
// =============================================================================

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    // Test memory scaling with witness size.
    for &num_elements in &[1024, 4096, 16384] {
        group.throughput(Throughput::Bytes((num_elements * 32) as u64)); // Fr is ~32 bytes

        group.bench_with_input(
            BenchmarkId::new("witness_allocation", num_elements),
            &num_elements,
            |b, &num_elements| {
                let mut rng = ChaCha20Rng::seed_from_u64(12345);

                b.iter(|| {
                    let witness = create_share_validity_witness(num_elements, &mut rng);
                    black_box(witness)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    // Share validity benchmarks
    bench_share_validity_proof_generation,
    bench_share_validity_proof_verification,
    bench_share_validity_serialization,
    // Aggregation benchmarks
    bench_aggregation_proof_generation,
    bench_aggregation_proof_verification,
    bench_aggregation_scaling,
    // MAC benchmarks
    bench_mac_proof_generation,
    bench_mac_proof_verification,
    // Batch benchmarks
    bench_batch_verification,
    bench_batch_mixed_proofs,
    // Full pipeline benchmarks
    bench_full_training_step,
    bench_proof_size,
    bench_memory_usage,
);

criterion_main!(benches);
