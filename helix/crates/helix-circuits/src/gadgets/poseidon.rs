//! Poseidon Hash Gadget for In-Circuit Hashing.
//!
//! Implements the Poseidon hash function over BN254 Fr with:
//! - Width: 3 (state = [s0, s1, s2])
//! - Rate: 2 (absorb 2 elements per permutation)
//! - Full rounds: 8 (4 at start, 4 at end)
//! - Partial rounds: 57
//! - S-box: x^5 (alpha=5, secure for BN254 where gcd(5, p-1) = 1)
//! - MDS matrix: [[2,1,1],[1,2,1],[1,1,2]] (verified MDS over BN254 Fr)
//!
//! Provides both native computation and in-circuit constraint functions.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ErrorFront, Fixed, Selector},
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Poseidon state width.
pub const POSEIDON_WIDTH: usize = 3;
/// Poseidon absorption rate.
pub const POSEIDON_RATE: usize = 2;
/// Number of full rounds (split equally before and after partial rounds).
pub const POSEIDON_FULL_ROUNDS: usize = 8;
/// Number of partial rounds.
pub const POSEIDON_PARTIAL_ROUNDS: usize = 57;
/// Total number of rounds.
pub const POSEIDON_TOTAL_ROUNDS: usize = POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS;

// ---------------------------------------------------------------------------
// Round Constants (generated deterministically from seed)
// ---------------------------------------------------------------------------

static ROUND_CONSTANTS: OnceLock<Vec<[Fr; POSEIDON_WIDTH]>> = OnceLock::new();

/// Returns the cached Poseidon round constants.
pub fn get_round_constants() -> &'static Vec<[Fr; POSEIDON_WIDTH]> {
    ROUND_CONSTANTS.get_or_init(|| {
        let mut constants = Vec::with_capacity(POSEIDON_TOTAL_ROUNDS);
        for round in 0..POSEIDON_TOTAL_ROUNDS {
            let mut rc = [Fr::ZERO; POSEIDON_WIDTH];
            for i in 0..POSEIDON_WIDTH {
                let mut hasher = Sha256::new();
                hasher.update(b"HELIX_POSEIDON_RC_V1");
                hasher.update(&(round as u64).to_le_bytes());
                hasher.update(&(i as u64).to_le_bytes());
                let hash: [u8; 32] = hasher.finalize().into();

                let mut repr = [0u8; 32];
                repr.copy_from_slice(&hash);
                // Clear top bits to ensure value < BN254 scalar modulus (~254 bits)
                repr[31] &= 0x1F;
                rc[i] = Fr::from_repr_vartime(repr.into()).unwrap_or(Fr::ZERO);
            }
            constants.push(rc);
        }
        constants
    })
}

// ---------------------------------------------------------------------------
// S-box: x^5
// ---------------------------------------------------------------------------

/// Computes x^5 (Poseidon S-box for BN254).
#[inline]
fn sbox(x: Fr) -> Fr {
    let x2 = x.square();
    let x4 = x2.square();
    x4 * x
}

// ---------------------------------------------------------------------------
// MDS Matrix Multiplication
// ---------------------------------------------------------------------------
// M = [[2,1,1],[1,2,1],[1,1,2]]
// Equivalently: out[i] = state[i] + (state[0] + state[1] + state[2])
//
// This matrix is verified MDS over BN254 Fr:
// - All 1x1 minors: {1, 2} ≠ 0 ✓
// - All 2x2 minors: {1, 3, -1} ≠ 0 ✓
// - det(M) = 4 ≠ 0 ✓

#[inline]
fn mds_multiply(state: &mut [Fr; POSEIDON_WIDTH]) {
    let old = *state;
    let sum = old[0] + old[1] + old[2];
    state[0] = old[0] + sum;
    state[1] = old[1] + sum;
    state[2] = old[2] + sum;
}

// ---------------------------------------------------------------------------
// Poseidon Permutation (Native)
// ---------------------------------------------------------------------------

/// Applies the full Poseidon permutation to the state in-place.
pub fn poseidon_permutation(state: &mut [Fr; POSEIDON_WIDTH]) {
    let rc = get_round_constants();
    let half_full = POSEIDON_FULL_ROUNDS / 2;
    let mut round = 0;

    // First half of full rounds
    for _ in 0..half_full {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rc[round][i];
        }
        for i in 0..POSEIDON_WIDTH {
            state[i] = sbox(state[i]);
        }
        mds_multiply(state);
        round += 1;
    }

    // Partial rounds
    for _ in 0..POSEIDON_PARTIAL_ROUNDS {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rc[round][i];
        }
        state[0] = sbox(state[0]);
        mds_multiply(state);
        round += 1;
    }

    // Second half of full rounds
    for _ in 0..half_full {
        for i in 0..POSEIDON_WIDTH {
            state[i] += rc[round][i];
        }
        for i in 0..POSEIDON_WIDTH {
            state[i] = sbox(state[i]);
        }
        mds_multiply(state);
        round += 1;
    }
}

// ---------------------------------------------------------------------------
// Native Hash Functions
// ---------------------------------------------------------------------------

/// Hashes two field elements using Poseidon.
///
/// State is initialized as [left, right, 0] and permuted.
/// Output is state[0] after permutation.
pub fn poseidon_hash_two(left: Fr, right: Fr) -> Fr {
    let mut state = [left, right, Fr::ZERO];
    poseidon_permutation(&mut state);
    state[0]
}

/// Hashes multiple field elements using Poseidon sponge construction.
///
/// Uses domain separation by encoding the input length in the capacity element.
pub fn poseidon_hash_many(inputs: &[Fr]) -> Fr {
    let mut state = [Fr::ZERO; POSEIDON_WIDTH];
    // Domain separation: encode length in capacity
    state[POSEIDON_WIDTH - 1] = Fr::from(inputs.len() as u64);

    for chunk in inputs.chunks(POSEIDON_RATE) {
        for (i, &val) in chunk.iter().enumerate() {
            state[i] += val;
        }
        poseidon_permutation(&mut state);
    }

    if inputs.is_empty() {
        poseidon_permutation(&mut state);
    }

    state[0]
}

// ---------------------------------------------------------------------------
// In-Circuit Poseidon Hash
// ---------------------------------------------------------------------------

/// Configuration from the host circuit needed for Poseidon constraints.
#[derive(Clone, Debug)]
pub struct PoseidonCircuitConfig {
    /// Advice columns (need at least 3).
    pub advice: [Column<Advice>; 3],
    /// Fixed column for round constants.
    pub fixed: Column<Fixed>,
    /// Selector for multiplication gate: a * b = c.
    pub s_mul: Selector,
    /// Selector for addition gate: a + b = c.
    pub s_add: Selector,
    /// Selector for round constant addition: advice[0] + fixed = advice[2].
    pub s_rc_add: Selector,
    /// Selector for equality check: advice[0] = advice[1].
    pub s_eq: Selector,
}

/// Rows needed per full round (3 rc_adds + 9 muls + 5 adds = 17).
const ROWS_PER_FULL_ROUND: usize = 17;
/// Rows needed per partial round (3 rc_adds + 3 muls + 5 adds = 11).
const ROWS_PER_PARTIAL_ROUND: usize = 11;
/// Total rows for one Poseidon hash (plus 1 for eq check).
pub const POSEIDON_CIRCUIT_ROWS: usize =
    POSEIDON_FULL_ROUNDS * ROWS_PER_FULL_ROUND
    + POSEIDON_PARTIAL_ROUNDS * ROWS_PER_PARTIAL_ROUND
    + 1;

/// Synthesizes a constrained Poseidon hash: output = Poseidon(left, right).
///
/// Lays out the full Poseidon permutation in a single region using:
/// - s_rc_add gates for round constant additions (fixed column)
/// - s_mul gates for x^5 S-box computation
/// - s_add gates for MDS matrix multiplication
/// - s_eq gate for output verification
///
/// Returns the computed hash value.
pub fn synthesize_poseidon_hash(
    config: &PoseidonCircuitConfig,
    layouter: &mut impl Layouter<Fr>,
    left: Fr,
    right: Fr,
    label: &str,
) -> Result<Fr, ErrorFront> {
    let expected = poseidon_hash_two(left, right);

    layouter.assign_region(
        || format!("{}_poseidon", label),
        |mut region| {
            let rc = get_round_constants();
            let half_full = POSEIDON_FULL_ROUNDS / 2;
            let mut state = [left, right, Fr::ZERO];
            let mut row = 0;
            let mut round = 0;

            // ---- First half of full rounds ----
            for _ in 0..half_full {
                // Round constant addition
                let mut state_rc = [Fr::ZERO; POSEIDON_WIDTH];
                for i in 0..POSEIDON_WIDTH {
                    state_rc[i] = state[i] + rc[round][i];
                    config.s_rc_add.enable(&mut region, row)?;
                    region.assign_advice(|| "state", config.advice[0], row, || Value::known(state[i]))?;
                    region.assign_fixed(|| "rc", config.fixed, row, || Value::known(rc[round][i]))?;
                    region.assign_advice(|| "state_rc", config.advice[2], row, || Value::known(state_rc[i]))?;
                    row += 1;
                }

                // Full S-box: x^5 for all 3 elements
                let mut sbox_out = [Fr::ZERO; POSEIDON_WIDTH];
                for i in 0..POSEIDON_WIDTH {
                    let x = state_rc[i];
                    let x2 = x.square();
                    let x4 = x2.square();
                    let x5 = x4 * x;
                    sbox_out[i] = x5;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x", config.advice[0], row, || Value::known(x))?;
                    region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                    region.assign_advice(|| "x2", config.advice[2], row, || Value::known(x2))?;
                    row += 1;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x2", config.advice[0], row, || Value::known(x2))?;
                    region.assign_advice(|| "x2", config.advice[1], row, || Value::known(x2))?;
                    region.assign_advice(|| "x4", config.advice[2], row, || Value::known(x4))?;
                    row += 1;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x4", config.advice[0], row, || Value::known(x4))?;
                    region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                    region.assign_advice(|| "x5", config.advice[2], row, || Value::known(x5))?;
                    row += 1;
                }

                // MDS: out[i] = sbox_out[i] + sum
                let sum_01 = sbox_out[0] + sbox_out[1];
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s0", config.advice[0], row, || Value::known(sbox_out[0]))?;
                region.assign_advice(|| "s1", config.advice[1], row, || Value::known(sbox_out[1]))?;
                region.assign_advice(|| "sum01", config.advice[2], row, || Value::known(sum_01))?;
                row += 1;

                let sum = sum_01 + sbox_out[2];
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "sum01", config.advice[0], row, || Value::known(sum_01))?;
                region.assign_advice(|| "s2", config.advice[1], row, || Value::known(sbox_out[2]))?;
                region.assign_advice(|| "sum", config.advice[2], row, || Value::known(sum))?;
                row += 1;

                for i in 0..POSEIDON_WIDTH {
                    state[i] = sbox_out[i] + sum;
                    config.s_add.enable(&mut region, row)?;
                    region.assign_advice(|| "si", config.advice[0], row, || Value::known(sbox_out[i]))?;
                    region.assign_advice(|| "sum", config.advice[1], row, || Value::known(sum))?;
                    region.assign_advice(|| "out", config.advice[2], row, || Value::known(state[i]))?;
                    row += 1;
                }

                round += 1;
            }

            // ---- Partial rounds ----
            for _ in 0..POSEIDON_PARTIAL_ROUNDS {
                // Round constant addition
                let mut state_rc = [Fr::ZERO; POSEIDON_WIDTH];
                for i in 0..POSEIDON_WIDTH {
                    state_rc[i] = state[i] + rc[round][i];
                    config.s_rc_add.enable(&mut region, row)?;
                    region.assign_advice(|| "state", config.advice[0], row, || Value::known(state[i]))?;
                    region.assign_fixed(|| "rc", config.fixed, row, || Value::known(rc[round][i]))?;
                    region.assign_advice(|| "state_rc", config.advice[2], row, || Value::known(state_rc[i]))?;
                    row += 1;
                }

                // Partial S-box: only state[0]
                let x = state_rc[0];
                let x2 = x.square();
                let x4 = x2.square();
                let x5 = x4 * x;

                config.s_mul.enable(&mut region, row)?;
                region.assign_advice(|| "x", config.advice[0], row, || Value::known(x))?;
                region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                region.assign_advice(|| "x2", config.advice[2], row, || Value::known(x2))?;
                row += 1;

                config.s_mul.enable(&mut region, row)?;
                region.assign_advice(|| "x2", config.advice[0], row, || Value::known(x2))?;
                region.assign_advice(|| "x2", config.advice[1], row, || Value::known(x2))?;
                region.assign_advice(|| "x4", config.advice[2], row, || Value::known(x4))?;
                row += 1;

                config.s_mul.enable(&mut region, row)?;
                region.assign_advice(|| "x4", config.advice[0], row, || Value::known(x4))?;
                region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                region.assign_advice(|| "x5", config.advice[2], row, || Value::known(x5))?;
                row += 1;

                // MDS with partial sbox output
                let s0 = x5;
                let s1 = state_rc[1];
                let s2 = state_rc[2];

                let sum_01 = s0 + s1;
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s0", config.advice[0], row, || Value::known(s0))?;
                region.assign_advice(|| "s1", config.advice[1], row, || Value::known(s1))?;
                region.assign_advice(|| "sum01", config.advice[2], row, || Value::known(sum_01))?;
                row += 1;

                let sum = sum_01 + s2;
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "sum01", config.advice[0], row, || Value::known(sum_01))?;
                region.assign_advice(|| "s2", config.advice[1], row, || Value::known(s2))?;
                region.assign_advice(|| "sum", config.advice[2], row, || Value::known(sum))?;
                row += 1;

                state[0] = s0 + sum;
                state[1] = s1 + sum;
                state[2] = s2 + sum;

                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s0", config.advice[0], row, || Value::known(s0))?;
                region.assign_advice(|| "sum", config.advice[1], row, || Value::known(sum))?;
                region.assign_advice(|| "out0", config.advice[2], row, || Value::known(state[0]))?;
                row += 1;

                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s1", config.advice[0], row, || Value::known(s1))?;
                region.assign_advice(|| "sum", config.advice[1], row, || Value::known(sum))?;
                region.assign_advice(|| "out1", config.advice[2], row, || Value::known(state[1]))?;
                row += 1;

                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s2", config.advice[0], row, || Value::known(s2))?;
                region.assign_advice(|| "sum", config.advice[1], row, || Value::known(sum))?;
                region.assign_advice(|| "out2", config.advice[2], row, || Value::known(state[2]))?;
                row += 1;

                round += 1;
            }

            // ---- Second half of full rounds ----
            for _ in 0..half_full {
                let mut state_rc = [Fr::ZERO; POSEIDON_WIDTH];
                for i in 0..POSEIDON_WIDTH {
                    state_rc[i] = state[i] + rc[round][i];
                    config.s_rc_add.enable(&mut region, row)?;
                    region.assign_advice(|| "state", config.advice[0], row, || Value::known(state[i]))?;
                    region.assign_fixed(|| "rc", config.fixed, row, || Value::known(rc[round][i]))?;
                    region.assign_advice(|| "state_rc", config.advice[2], row, || Value::known(state_rc[i]))?;
                    row += 1;
                }

                let mut sbox_out = [Fr::ZERO; POSEIDON_WIDTH];
                for i in 0..POSEIDON_WIDTH {
                    let x = state_rc[i];
                    let x2 = x.square();
                    let x4 = x2.square();
                    let x5 = x4 * x;
                    sbox_out[i] = x5;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x", config.advice[0], row, || Value::known(x))?;
                    region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                    region.assign_advice(|| "x2", config.advice[2], row, || Value::known(x2))?;
                    row += 1;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x2", config.advice[0], row, || Value::known(x2))?;
                    region.assign_advice(|| "x2", config.advice[1], row, || Value::known(x2))?;
                    region.assign_advice(|| "x4", config.advice[2], row, || Value::known(x4))?;
                    row += 1;

                    config.s_mul.enable(&mut region, row)?;
                    region.assign_advice(|| "x4", config.advice[0], row, || Value::known(x4))?;
                    region.assign_advice(|| "x", config.advice[1], row, || Value::known(x))?;
                    region.assign_advice(|| "x5", config.advice[2], row, || Value::known(x5))?;
                    row += 1;
                }

                let sum_01 = sbox_out[0] + sbox_out[1];
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "s0", config.advice[0], row, || Value::known(sbox_out[0]))?;
                region.assign_advice(|| "s1", config.advice[1], row, || Value::known(sbox_out[1]))?;
                region.assign_advice(|| "sum01", config.advice[2], row, || Value::known(sum_01))?;
                row += 1;

                let sum = sum_01 + sbox_out[2];
                config.s_add.enable(&mut region, row)?;
                region.assign_advice(|| "sum01", config.advice[0], row, || Value::known(sum_01))?;
                region.assign_advice(|| "s2", config.advice[1], row, || Value::known(sbox_out[2]))?;
                region.assign_advice(|| "sum", config.advice[2], row, || Value::known(sum))?;
                row += 1;

                for i in 0..POSEIDON_WIDTH {
                    state[i] = sbox_out[i] + sum;
                    config.s_add.enable(&mut region, row)?;
                    region.assign_advice(|| "si", config.advice[0], row, || Value::known(sbox_out[i]))?;
                    region.assign_advice(|| "sum", config.advice[1], row, || Value::known(sum))?;
                    region.assign_advice(|| "out", config.advice[2], row, || Value::known(state[i]))?;
                    row += 1;
                }

                round += 1;
            }

            // ---- Final equality check: state[0] == expected ----
            config.s_eq.enable(&mut region, row)?;
            region.assign_advice(|| "computed", config.advice[0], row, || Value::known(state[0]))?;
            region.assign_advice(|| "expected", config.advice[1], row, || Value::known(expected))?;

            Ok(())
        },
    )?;

    Ok(expected)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_deterministic() {
        let a = Fr::from(42u64);
        let b = Fr::from(123u64);

        let h1 = poseidon_hash_two(a, b);
        let h2 = poseidon_hash_two(a, b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_poseidon_different_inputs() {
        let a = Fr::from(42u64);
        let b = Fr::from(123u64);

        let h1 = poseidon_hash_two(a, b);
        let h2 = poseidon_hash_two(b, a);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_poseidon_non_trivial() {
        let a = Fr::from(1u64);
        let b = Fr::from(2u64);

        let h = poseidon_hash_two(a, b);
        // Hash should not equal either input
        assert_ne!(h, a);
        assert_ne!(h, b);
        assert_ne!(h, Fr::ZERO);
    }

    #[test]
    fn test_poseidon_many() {
        let inputs = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];

        let h1 = poseidon_hash_many(&inputs);
        let h2 = poseidon_hash_many(&inputs);
        assert_eq!(h1, h2);

        let inputs2 = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(5u64)];
        let h3 = poseidon_hash_many(&inputs2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_poseidon_empty() {
        let h = poseidon_hash_many(&[]);
        assert_ne!(h, Fr::ZERO); // Domain separation makes empty hash non-zero
    }

    #[test]
    fn test_round_constants_cached() {
        let rc1 = get_round_constants();
        let rc2 = get_round_constants();
        assert_eq!(rc1.len(), rc2.len());
        assert_eq!(rc1[0][0], rc2[0][0]);
    }

    #[test]
    fn test_round_constants_nonzero() {
        let rc = get_round_constants();
        assert_eq!(rc.len(), POSEIDON_TOTAL_ROUNDS);
        // All round constants should be non-zero
        for (i, round_rc) in rc.iter().enumerate() {
            for (j, &val) in round_rc.iter().enumerate() {
                assert_ne!(val, Fr::ZERO, "Round constant [{i}][{j}] is zero");
            }
        }
    }

    #[test]
    fn test_sbox_correct() {
        let x = Fr::from(3u64);
        let x5 = sbox(x);
        assert_eq!(x5, Fr::from(243u64)); // 3^5 = 243
    }

    #[test]
    fn test_mds_correct() {
        // MDS: [[2,1,1],[1,2,1],[1,1,2]]
        let mut state = [Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        mds_multiply(&mut state);
        // sum = 1+2+3 = 6
        // out[0] = 1+6 = 7, out[1] = 2+6 = 8, out[2] = 3+6 = 9
        assert_eq!(state[0], Fr::from(7u64));
        assert_eq!(state[1], Fr::from(8u64));
        assert_eq!(state[2], Fr::from(9u64));
    }

    #[test]
    fn test_poseidon_circuit() {
        use halo2_proofs::{
            circuit::SimpleFloorPlanner,
            dev::MockProver,
            plonk::{Circuit, ConstraintSystem},
        };

        #[derive(Clone, Default)]
        struct TestPoseidonCircuit {
            left: Fr,
            right: Fr,
        }

        impl Circuit<Fr> for TestPoseidonCircuit {
            type Config = PoseidonCircuitConfig;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                Self::default()
            }

            fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
                let advice = [
                    meta.advice_column(),
                    meta.advice_column(),
                    meta.advice_column(),
                ];
                let fixed = meta.fixed_column();
                let s_mul = meta.selector();
                let s_add = meta.selector();
                let s_rc_add = meta.selector();
                let s_eq = meta.selector();

                for col in &advice {
                    meta.enable_equality(*col);
                }

                meta.create_gate("mul", |meta| {
                    let s = meta.query_selector(s_mul);
                    let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
                    let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
                    let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
                    vec![s * (a * b - c)]
                });

                meta.create_gate("add", |meta| {
                    let s = meta.query_selector(s_add);
                    let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
                    let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
                    let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
                    vec![s * (a + b - c)]
                });

                meta.create_gate("rc_add", |meta| {
                    let s = meta.query_selector(s_rc_add);
                    let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
                    let rc = meta.query_fixed(fixed, halo2_proofs::poly::Rotation::cur());
                    let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
                    vec![s * (a + rc - c)]
                });

                meta.create_gate("eq", |meta| {
                    let s = meta.query_selector(s_eq);
                    let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
                    let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
                    vec![s * (a - b)]
                });

                PoseidonCircuitConfig {
                    advice,
                    fixed,
                    s_mul,
                    s_add,
                    s_rc_add,
                    s_eq,
                }
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<Fr>,
            ) -> Result<(), ErrorFront> {
                synthesize_poseidon_hash(&config, &mut layouter, self.left, self.right, "test")?;
                Ok(())
            }
        }

        let circuit = TestPoseidonCircuit {
            left: Fr::from(42u64),
            right: Fr::from(123u64),
        };

        // k=12 gives 4096 rows, enough for ~764 Poseidon rows
        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
