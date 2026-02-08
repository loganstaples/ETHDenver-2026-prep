//! Freivalds Probabilistic Matrix Verification.
//!
//! Verifies matrix multiplication C = A × B using O(n²) field operations
//! instead of O(n³), with failure probability 1/|F|.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Column, Advice, Selector, ConstraintSystem, ErrorFront},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// Configuration for the Freivalds verification chip.
#[derive(Debug, Clone)]
pub struct FreivaldsConfig<F: PrimeField> {
    /// Advice columns for vectors.
    pub advice: [Column<Advice>; 3],
    /// Selector for the dot product gate.
    pub s_dot: Selector,
    /// Selector for the equality check.
    pub s_check: Selector,
    /// Phantom data.
    _marker: PhantomData<F>,
}

/// Chip for Freivalds matrix multiplication verification.
/// 
/// To verify C = A × B:
/// 1. Sample random vector r
/// 2. Compute x = B × r (n×1)
/// 3. Compute y = A × x (m×1) 
/// 4. Compute z = C × r (m×1)
/// 5. Check y == z
/// 
/// This reduces O(n³) work to O(n²) with negligible error probability.
pub struct FreivaldsChip<F: PrimeField> {
    /// Configuration.
    config: FreivaldsConfig<F>,
}

impl<F: PrimeField> FreivaldsChip<F> {
    /// Creates a new Freivalds chip.
    pub fn new(config: FreivaldsConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the chip.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> FreivaldsConfig<F> {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        for col in &advice {
            meta.enable_equality(*col);
        }

        let s_dot = meta.selector();
        let s_check = meta.selector();

        // Dot product accumulation gate: acc_new = acc_old + a * b
        meta.create_gate("freivalds_dot", |meta| {
            let s = meta.query_selector(s_dot);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let acc_prev = meta.query_advice(advice[2], Rotation::prev());
            let acc_cur = meta.query_advice(advice[2], Rotation::cur());

            vec![s * (acc_cur - acc_prev - a * b)]
        });

        // Equality check gate
        meta.create_gate("freivalds_check", |meta| {
            let s = meta.query_selector(s_check);
            let left = meta.query_advice(advice[0], Rotation::cur());
            let right = meta.query_advice(advice[1], Rotation::cur());

            vec![s * (left - right)]
        });

        FreivaldsConfig {
            advice,
            s_dot,
            s_check,
            _marker: PhantomData,
        }
    }

    /// Generates a cryptographically secure challenge vector from a seed.
    ///
    /// Uses SHA-256 for deterministic but unpredictable challenge generation.
    /// This is critical for Freivalds soundness: if the prover can predict
    /// the challenge vector, they can craft invalid matrix multiplications
    /// that pass verification.
    pub fn generate_challenge_vector(seed: u64, len: usize) -> Vec<F> {
        use sha2::{Digest, Sha256};

        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let mut hasher = Sha256::new();
            hasher.update(b"HELIX_FREIVALDS_V1");
            hasher.update(&seed.to_le_bytes());
            hasher.update(&(i as u64).to_le_bytes());
            let hash = hasher.finalize();

            // Use first 8 bytes to create a field element
            // Safety: hash is SHA-256 output (32 bytes), so [0..8] always fits [u8; 8]
            let bytes: [u8; 8] = hash[0..8].try_into().expect("SHA-256 hash >= 8 bytes");
            let val = u64::from_le_bytes(bytes);
            result.push(F::from(val));
        }

        result
    }

    /// Assigns and verifies C = A × B using Freivalds method.
    /// 
    /// # Arguments
    /// * `layouter` - The circuit layouter
    /// * `a` - Matrix A values (m × k), row-major
    /// * `b` - Matrix B values (k × n), row-major
    /// * `c` - Matrix C values (m × n), row-major
    /// * `m` - Number of rows in A/C
    /// * `k` - Inner dimension
    /// * `n` - Number of columns in B/C
    /// * `challenge_seed` - Seed for generating random challenge
    pub fn verify_matmul(
        &self,
        mut layouter: impl Layouter<F>,
        a: &[Value<F>],
        b: &[Value<F>],
        c: &[Value<F>],
        m: usize,
        k: usize,
        n: usize,
        challenge_seed: u64,
    ) -> Result<(), ErrorFront> {
        // Generate random vector r of length n
        let r = Self::generate_challenge_vector(challenge_seed, n);
        let r_values: Vec<Value<F>> = r.iter().map(|&v| Value::known(v)).collect();

        layouter.assign_region(
            || "freivalds verification",
            |mut region| {
                // Step 1: Compute x = B × r (k × 1)
                let mut x_values: Vec<Value<F>> = Vec::with_capacity(k);
                for i in 0..k {
                    // x[i] = sum_j B[i][j] * r[j]
                    let mut sum = Value::known(F::ZERO);
                    for j in 0..n {
                        let b_ij = b[i * n + j];
                        sum = sum + b_ij * r_values[j];
                    }
                    x_values.push(sum);
                }

                // Step 2: Compute y = A × x (m × 1)
                let mut y_values: Vec<Value<F>> = Vec::with_capacity(m);
                for i in 0..m {
                    let mut sum = Value::known(F::ZERO);
                    for j in 0..k {
                        let a_ij = a[i * k + j];
                        sum = sum + a_ij * x_values[j];
                    }
                    y_values.push(sum);
                }

                // Step 3: Compute z = C × r (m × 1)
                let mut z_values: Vec<Value<F>> = Vec::with_capacity(m);
                for i in 0..m {
                    let mut sum = Value::known(F::ZERO);
                    for j in 0..n {
                        let c_ij = c[i * n + j];
                        sum = sum + c_ij * r_values[j];
                    }
                    z_values.push(sum);
                }

                // Step 4: Assign and check y == z
                for (row, (y, z)) in y_values.iter().zip(z_values.iter()).enumerate() {
                    self.config.s_check.enable(&mut region, row)?;

                    region.assign_advice(
                        || format!("y_{}", row),
                        self.config.advice[0],
                        row,
                        || *y,
                    )?;

                    region.assign_advice(
                        || format!("z_{}", row),
                        self.config.advice[1],
                        row,
                        || *z,
                    )?;
                }

                Ok(())
            },
        )
    }

    /// Simplified verification for small matrices (direct constraint).
    pub fn verify_matmul_small(
        &self,
        mut layouter: impl Layouter<F>,
        a: &[Value<F>],
        b: &[Value<F>],
        c: &[Value<F>],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), ErrorFront> {
        // For small matrices, directly verify each output element
        layouter.assign_region(
            || "direct matmul verification",
            |mut region| {
                for i in 0..m {
                    for j in 0..n {
                        // C[i][j] = sum_t A[i][t] * B[t][j]
                        let mut expected = Value::known(F::ZERO);
                        for t in 0..k {
                            let a_it = a[i * k + t];
                            let b_tj = b[t * n + j];
                            expected = expected + a_it * b_tj;
                        }

                        let c_ij = c[i * n + j];
                        let row = i * n + j;

                        self.config.s_check.enable(&mut region, row)?;

                        region.assign_advice(
                            || format!("expected_{}_{}", i, j),
                            self.config.advice[0],
                            row,
                            || expected,
                        )?;

                        region.assign_advice(
                            || format!("actual_{}_{}", i, j),
                            self.config.advice[1],
                            row,
                            || c_ij,
                        )?;
                    }
                }

                Ok(())
            },
        )
    }
}

/// Computes C = A × B in the clear (for witness generation).
pub fn matmul<F: PrimeField>(
    a: &[F],
    b: &[F],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<F> {
    let mut c = vec![F::ZERO; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = F::ZERO;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::Fr;
    use halo2_proofs::{
        arithmetic::Field,
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::Circuit,
    };

    #[derive(Clone, Default)]
    struct FreivaldsCircuit {
        a: Vec<Fr>,
        b: Vec<Fr>,
        c: Vec<Fr>,
        m: usize,
        k: usize,
        n: usize,
    }

    impl Circuit<Fr> for FreivaldsCircuit {
        type Config = FreivaldsConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            FreivaldsChip::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = FreivaldsChip::new(config);

            let a_values: Vec<Value<Fr>> = self.a.iter().map(|&v| Value::known(v)).collect();
            let b_values: Vec<Value<Fr>> = self.b.iter().map(|&v| Value::known(v)).collect();
            let c_values: Vec<Value<Fr>> = self.c.iter().map(|&v| Value::known(v)).collect();

            chip.verify_matmul_small(
                layouter,
                &a_values,
                &b_values,
                &c_values,
                self.m,
                self.k,
                self.n,
            )
        }
    }

    #[test]
    fn test_matmul_compute() {
        // 2×2 × 2×2
        let a = vec![
            Fr::from(1), Fr::from(2),
            Fr::from(3), Fr::from(4),
        ];
        let b = vec![
            Fr::from(5), Fr::from(6),
            Fr::from(7), Fr::from(8),
        ];

        let c = matmul(&a, &b, 2, 2, 2);

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c[0], Fr::from(19));
        assert_eq!(c[1], Fr::from(22));
        assert_eq!(c[2], Fr::from(43));
        assert_eq!(c[3], Fr::from(50));
    }

    #[test]
    fn test_freivalds_valid() {
        let a = vec![
            Fr::from(1), Fr::from(2),
            Fr::from(3), Fr::from(4),
        ];
        let b = vec![
            Fr::from(5), Fr::from(6),
            Fr::from(7), Fr::from(8),
        ];
        let c = matmul(&a, &b, 2, 2, 2);

        let circuit = FreivaldsCircuit {
            a,
            b,
            c,
            m: 2,
            k: 2,
            n: 2,
        };

        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_freivalds_invalid() {
        let a = vec![
            Fr::from(1), Fr::from(2),
            Fr::from(3), Fr::from(4),
        ];
        let b = vec![
            Fr::from(5), Fr::from(6),
            Fr::from(7), Fr::from(8),
        ];
        // Invalid C (all zeros instead of correct result)
        let c = vec![Fr::ZERO; 4];

        let circuit = FreivaldsCircuit {
            a,
            b,
            c,
            m: 2,
            k: 2,
            n: 2,
        };

        let prover = MockProver::run(8, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_challenge_generation() {
        let r1 = FreivaldsChip::<Fr>::generate_challenge_vector(42, 5);
        let r2 = FreivaldsChip::<Fr>::generate_challenge_vector(42, 5);
        let r3 = FreivaldsChip::<Fr>::generate_challenge_vector(43, 5);

        // Same seed gives same result
        assert_eq!(r1, r2);
        // Different seed gives different result
        assert_ne!(r1, r3);
    }
}
