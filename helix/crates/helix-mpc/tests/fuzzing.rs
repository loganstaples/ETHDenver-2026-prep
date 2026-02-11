//! Property-based and fuzzing tests for helix-mpc.
//!
//! Validates fundamental invariants across many random inputs:
//! - Secret sharing roundtrip
//! - Beaver triple correctness
//! - Field arithmetic distributivity
//! - Circuit bridge fixed-point encoding roundtrip

use helix_mpc::field::Fr;
use helix_mpc::beaver::dealer::TrustedDealer;
use helix_mpc::beaver::distributed::DistributedTripleGen;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Split a value into n additive shares in Fr (raw field elements).
fn split_fr(value: &Fr, n: usize, rng: &mut impl rand::RngCore) -> Vec<Fr> {
    let mut shares = Vec::with_capacity(n);
    let mut sum = Fr::ZERO;
    for _ in 0..n - 1 {
        let r = Fr::random(rng);
        sum = Fr::add(&sum, &r);
        shares.push(r);
    }
    shares.push(Fr::sub(value, &sum));
    shares
}

/// Split a fixed-point value into n additive shares that are also
/// fixed-point encoded. This is required for Beaver multiplication
/// which uses fixed_mul internally.
fn split_fr_fixedpoint(value: &Fr, n: usize, rng: &mut (impl rand::RngCore + Rng)) -> Vec<Fr> {
    let mut shares = Vec::with_capacity(n);
    let mut sum = Fr::ZERO;
    for _ in 0..n - 1 {
        let r = Fr::from_f64(rng.gen_range(-1000.0..1000.0));
        sum = Fr::add(&sum, &r);
        shares.push(r);
    }
    shares.push(Fr::sub(value, &sum));
    shares
}

/// Reconstruct from additive shares by summing.
fn reconstruct_fr(shares: &[Fr]) -> Fr {
    shares.iter().fold(Fr::ZERO, |acc, s| Fr::add(&acc, s))
}

// =============================================================================
// Secret Sharing Roundtrip
// =============================================================================

#[test]
fn test_sharing_roundtrip_10000() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    for n in [2, 3, 5, 7] {
        for trial in 0..2500 {
            let original = Fr::random(&mut rng);
            let shares = split_fr(&original, n, &mut rng);
            let reconstructed = reconstruct_fr(&shares);

            assert!(
                original.ct_eq(&reconstructed).to_bool(),
                "Sharing roundtrip failed for n={}, trial={}",
                n,
                trial,
            );
        }
    }
}

// =============================================================================
// Beaver Triple Correctness: TrustedDealer
// =============================================================================

#[test]
fn test_beaver_triple_correctness_trusted_dealer_10000() {
    let mut dealer = TrustedDealer::with_seed(123);
    let num_parties = 3;

    for _ in 0..100 {
        let batch = dealer.generate_scalar_triples(100, num_parties);

        for triple_idx in 0..100 {
            // Reconstruct a, b, c from all parties' shares.
            let a: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));

            // TrustedDealer uses mpc_scale (field-exact fixed-point scaling).
            let ab = a.mpc_scale(&b);
            assert!(
                ab.ct_eq(&c).to_bool(),
                "Beaver triple a*b != c for triple {}",
                triple_idx,
            );
        }
    }
}

// =============================================================================
// Beaver Triple Correctness: Distributed
// =============================================================================

#[test]
fn test_beaver_triple_correctness_distributed_1000() {
    let num_parties = 3;

    for seed in 0..100 {
        let batch = DistributedTripleGen::simulate_distributed_batch(10, num_parties, seed);

        for triple_idx in 0..10 {
            let a: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].a.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let b: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].b.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));
            let c: Fr = (0..num_parties)
                .map(|p| batch[p][triple_idx].c.clone())
                .fold(Fr::ZERO, |acc, v| Fr::add(&acc, &v));

            let ab = a.mpc_scale(&b);
            assert!(
                ab.ct_eq(&c).to_bool(),
                "Distributed triple a*b != c for seed={}, idx={}",
                seed,
                triple_idx,
            );
        }
    }
}

// =============================================================================
// Field Arithmetic Properties
// =============================================================================

#[test]
fn test_field_distributivity_10000() {
    let mut rng = ChaCha20Rng::seed_from_u64(99);

    for _ in 0..10000 {
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);
        let c = Fr::random(&mut rng);

        // (a + b) * c == a*c + b*c
        let lhs = Fr::mul(&Fr::add(&a, &b), &c);
        let rhs = Fr::add(&Fr::mul(&a, &c), &Fr::mul(&b, &c));

        assert!(
            lhs.ct_eq(&rhs).to_bool(),
            "Distributivity failed",
        );
    }
}

#[test]
fn test_field_commutativity_10000() {
    let mut rng = ChaCha20Rng::seed_from_u64(77);

    for _ in 0..10000 {
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);

        // a + b == b + a
        assert!(Fr::add(&a, &b).ct_eq(&Fr::add(&b, &a)).to_bool());
        // a * b == b * a
        assert!(Fr::mul(&a, &b).ct_eq(&Fr::mul(&b, &a)).to_bool());
    }
}

#[test]
fn test_field_additive_inverse_10000() {
    let mut rng = ChaCha20Rng::seed_from_u64(55);

    for _ in 0..10000 {
        let a = Fr::random(&mut rng);
        let neg_a = Fr::sub(&Fr::ZERO, &a);
        let result = Fr::add(&a, &neg_a);

        assert!(
            result.ct_eq(&Fr::ZERO).to_bool(),
            "a + (-a) != 0",
        );
    }
}

#[test]
fn test_field_associativity_10000() {
    let mut rng = ChaCha20Rng::seed_from_u64(33);

    for _ in 0..10000 {
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);
        let c = Fr::random(&mut rng);

        // (a + b) + c == a + (b + c)
        let lhs_add = Fr::add(&Fr::add(&a, &b), &c);
        let rhs_add = Fr::add(&a, &Fr::add(&b, &c));
        assert!(lhs_add.ct_eq(&rhs_add).to_bool());

        // (a * b) * c == a * (b * c)
        let lhs_mul = Fr::mul(&Fr::mul(&a, &b), &c);
        let rhs_mul = Fr::mul(&a, &Fr::mul(&b, &c));
        assert!(lhs_mul.ct_eq(&rhs_mul).to_bool());
    }
}

// =============================================================================
// Fixed-point Encoding Roundtrip
// =============================================================================

#[test]
fn test_fixed_point_roundtrip_range() {
    // Test roundtrip for values in typical neural network range.
    let values: Vec<f64> = (-1000..=1000)
        .map(|i| i as f64 * 0.01)
        .collect();

    for &v in &values {
        let fr = Fr::from_f64(v);
        let back = fr.to_f64();
        assert!(
            (back - v).abs() < 1e-6,
            "Fixed-point roundtrip failed for {}: got {}",
            v,
            back,
        );
    }
}

#[test]
fn test_fixed_point_extreme_values() {
    for &v in &[1e10, -1e10, 1e12, -1e12, 0.0, 1.0, -1.0] {
        let fr = Fr::from_f64(v);
        let back = fr.to_f64();
        let rel_err = if v.abs() > 1.0 {
            (back - v).abs() / v.abs()
        } else {
            (back - v).abs()
        };
        assert!(
            rel_err < 1e-6,
            "Fixed-point extreme value failed for {}: got {}, rel_err={}",
            v,
            back,
            rel_err,
        );
    }
}

// =============================================================================
// Sharing + Beaver Multiplication E2E
// =============================================================================

#[test]
fn test_shared_multiplication_1000() {
    // Full multiplication protocol: share a, share b, use beaver triple
    // to compute shares of a*b, reconstruct and verify.
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut dealer = TrustedDealer::with_seed(42);
    let num_parties = 3;

    for trial in 0..1000 {
        // Use fixed-point encoded values (not random field elements)
        // since TrustedDealer uses fixed_mul which expects the 2^64 scaling.
        let x_val = (trial as f64 * 0.037 - 18.5).sin() * 10.0;
        let y_val = (trial as f64 * 0.071 + 3.14).cos() * 10.0;
        let x = Fr::from_f64(x_val);
        let y = Fr::from_f64(y_val);

        // Share x and y using fixed-point shares (required for fixed_mul protocol).
        let x_shares = split_fr_fixedpoint(&x, num_parties, &mut rng);
        let y_shares = split_fr_fixedpoint(&y, num_parties, &mut rng);

        // Get beaver triple
        let triples = dealer.generate_scalar_triples(1, num_parties);

        // Beaver multiplication protocol
        // d_i = x_i - a_i, e_i = y_i - b_i
        let mut d_shares = Vec::with_capacity(num_parties);
        let mut e_shares = Vec::with_capacity(num_parties);
        for i in 0..num_parties {
            d_shares.push(Fr::sub(&x_shares[i], &triples[i][0].a));
            e_shares.push(Fr::sub(&y_shares[i], &triples[i][0].b));
        }

        // Open d and e
        let d = reconstruct_fr(&d_shares);
        let e = reconstruct_fr(&e_shares);

        // z_i = c_i + d*b_i + e*a_i + (i==0 ? d*e : 0)
        // TrustedDealer uses mpc_scale, so the Beaver protocol must too.
        let mut z_shares = Vec::with_capacity(num_parties);
        for i in 0..num_parties {
            let mut z = triples[i][0].c.clone();
            z = Fr::add(&z, &d.mpc_scale(&triples[i][0].b));
            z = Fr::add(&z, &e.mpc_scale(&triples[i][0].a));
            if i == 0 {
                z = Fr::add(&z, &d.mpc_scale(&e));
            }
            z_shares.push(z);
        }

        let xy = x.mpc_scale(&y);
        let reconstructed = reconstruct_fr(&z_shares);

        // Compare in f64 domain with tolerance since shares are fixed-point
        // encoded and the protocol introduces small rounding.
        let expected = xy.to_f64();
        let actual = reconstructed.to_f64();
        assert!(
            (expected - actual).abs() < 1e-3,
            "Shared multiplication failed at trial {}: expected {}, got {} (diff {})",
            trial, expected, actual, (expected - actual).abs(),
        );
    }
}
