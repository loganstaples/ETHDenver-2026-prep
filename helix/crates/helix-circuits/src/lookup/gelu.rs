//! GELU Lookup Table Implementation.
//!
//! This module provides production-grade GELU (Gaussian Error Linear Unit)
//! lookup tables for transformer models and modern neural networks.
//!
//! # GELU Definition
//!
//! GELU(x) = x * Φ(x)
//!
//! where Φ(x) is the standard normal CDF. This is commonly approximated as:
//!
//! GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//!
//! # Performance
//!
//! Lookup-based GELU provides massive speedup over arithmetic circuits:
//! - **1 constraint** for lookup vs **50+ constraints** for polynomial approximation
//! - **Essential** for efficient transformer verification
//!
//! # Accuracy
//!
//! With INT8 quantization and scale=64:
//! - Maximum error: < 0.02 (relative to unscaled values)
//! - Suitable for inference and approximate training verification

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Column, ConstraintSystem, ErrorFront, Selector},
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

use super::table::{PlookupTable, PlookupChip, PlookupTableConfig};

/// GELU approximation constant: √(2/π).
const SQRT_2_PI: f64 = 0.7978845608028654;

/// GELU approximation constant for cubic term.
const GELU_CUBIC_COEFF: f64 = 0.044715;

/// Configuration for GELU lookup.
#[derive(Clone, Debug)]
pub struct GELULookupConfig<F: PrimeField> {
    /// Underlying plookup configuration.
    pub inner: PlookupTableConfig<F>,
    /// Selector specifically for GELU.
    pub s_gelu: Selector,
}

/// GELU lookup table with configurable range and scale.
///
/// Supports efficient verification of GELU activations used in transformers.
#[derive(Clone)]
pub struct GELULookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    /// Precomputed table entries.
    table: PlookupTable<F>,
    /// Maximum approximation error.
    max_error: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> GELULookup<F, RANGE, SCALE> {
    /// Creates a new GELU lookup table.
    pub fn new() -> Self {
        let (table, max_error) = Self::build_table();
        Self {
            table,
            max_error,
            _marker: PhantomData,
        }
    }

    /// Builds the GELU lookup table.
    fn build_table() -> (PlookupTable<F>, f64) {
        let entries = gelu_table_entries::<F>(RANGE, SCALE);
        let mut table = PlookupTable::new(SCALE);

        let mut max_error = 0.0f64;
        let scale_f = SCALE as f64;
        let half = RANGE / 2;

        for (i, (input, output)) in entries.iter().enumerate() {
            table.add_entry(vec![*input], vec![*output]);

            // Calculate error for this entry using the known quantization
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            let exact = gelu_exact(x);
            let quantized_int = (exact * scale_f).round() as i64;
            let quantized = quantized_int as f64 / scale_f;
            let error = (exact - quantized).abs();
            max_error = max_error.max(error);
        }

        table.set_error_bound(F::from((max_error * scale_f).ceil() as u64 + 1));
        (table, max_error)
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Returns the entries for loading.
    pub fn entries(&self) -> Vec<(F, F)> {
        self.table.two_column_entries()
    }

    /// Returns the maximum approximation error.
    pub fn max_error(&self) -> f64 {
        self.max_error
    }

    /// Computes GELU for a floating-point value.
    pub fn compute_float(x: f64) -> f64 {
        gelu_exact(x)
    }

    /// Computes GELU for a quantized value.
    pub fn compute_quantized(x: i64) -> i64 {
        let x_f = x as f64 / SCALE as f64;
        let gelu = gelu_exact(x_f);
        (gelu * SCALE as f64).round() as i64
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for GELULookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes exact GELU using the tanh approximation.
fn gelu_exact(x: f64) -> f64 {
    0.5 * x * (1.0 + (SQRT_2_PI * (x + GELU_CUBIC_COEFF * x.powi(3))).tanh())
}

/// GELU circuit chip for performing lookups.
pub struct GELUChip<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    config: GELULookupConfig<F>,
    lookup: GELULookup<F, RANGE, SCALE>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> GELUChip<F, RANGE, SCALE> {
    /// Creates a new GELU chip.
    pub fn new(config: GELULookupConfig<F>) -> Self {
        Self {
            config,
            lookup: GELULookup::new(),
        }
    }

    /// Configures the GELU lookup in the constraint system.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> GELULookupConfig<F> {
        let inner = PlookupChip::configure_simple(meta, advice_input, advice_output);
        GELULookupConfig {
            s_gelu: inner.s_lookup,
            inner,
        }
    }

    /// Loads the GELU lookup table.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.load_table(layouter, &self.lookup.table)
    }

    /// Loads the GELU lookup table with padding.
    pub fn load_padded(&self, layouter: &mut impl Layouter<F>, min_rows: usize) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.load_table_padded(layouter, &self.lookup.table, min_rows)
    }

    /// Performs a GELU lookup.
    pub fn gelu(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.lookup_single(region, row, input, output)
    }

    /// Performs GELU and returns the assigned output cell.
    pub fn gelu_with_output(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        input: Value<F>,
    ) -> Result<AssignedCell<F, F>, ErrorFront> {
        self.config.s_gelu.enable(region, row)?;

        region.assign_advice(
            || format!("gelu_input_{}", row),
            self.config.inner.advice_inputs[0],
            row,
            || input,
        )?;

        // Compute output by looking up in table
        let output = input.and_then(|x| {
            match self.lookup.table.lookup(&[x]) {
                Some(v) => Value::known(v[0]),
                None => Value::unknown(),
            }
        });

        region.assign_advice(
            || format!("gelu_output_{}", row),
            self.config.inner.advice_outputs[0],
            row,
            || output,
        )
    }

    /// Returns the configuration.
    pub fn config(&self) -> &GELULookupConfig<F> {
        &self.config
    }

    /// Returns the maximum error of this GELU approximation.
    pub fn max_error(&self) -> f64 {
        self.lookup.max_error()
    }
}

/// Generates GELU table entries for a given range and scale.
pub fn gelu_table_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let half = range / 2;
    let scale_f = scale as f64;
    let mut entries = Vec::with_capacity(range);

    for i in 0..range {
        let x = if i < half {
            i as f64 / scale_f
        } else {
            -((range - i) as f64) / scale_f
        };

        let gelu = gelu_exact(x);
        let quantized = (gelu * scale_f).round() as i64;

        let input_field = if i < half {
            F::from(i as u64)
        } else {
            F::ZERO - F::from((range - i) as u64)
        };

        let output_field = if quantized >= 0 {
            F::from(quantized as u64)
        } else {
            F::ZERO - F::from((-quantized) as u64)
        };

        entries.push((input_field, output_field));
    }

    entries
}

// ---------------------------------------------------------------------------
// Fast GELU Lookup (Simplified Approximation)
// ---------------------------------------------------------------------------

/// Fast GELU lookup using a simpler approximation.
///
/// FastGELU(x) ≈ x * sigmoid(1.702 * x)
///
/// This is faster to compute but slightly less accurate than standard GELU.
#[derive(Clone)]
pub struct FastGELULookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    max_error: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> FastGELULookup<F, RANGE, SCALE> {
    /// FastGELU constant.
    const FAST_GELU_COEFF: f64 = 1.702;

    /// Creates a new FastGELU lookup table.
    pub fn new() -> Self {
        let (table, max_error) = Self::build_table();
        Self {
            table,
            max_error,
            _marker: PhantomData,
        }
    }

    fn build_table() -> (PlookupTable<F>, f64) {
        let half = RANGE / 2;
        let scale_f = SCALE as f64;
        let mut table = PlookupTable::new(SCALE);
        let mut max_error = 0.0f64;

        for i in 0..RANGE {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            let fast_gelu = x * sigmoid(Self::FAST_GELU_COEFF * x);
            let quantized = (fast_gelu * scale_f).round() as i64;

            // Compare to exact GELU for error tracking
            let exact = gelu_exact(x);
            let error = (exact - fast_gelu).abs();
            max_error = max_error.max(error);

            let input_field = if i < half {
                F::from(i as u64)
            } else {
                F::ZERO - F::from((RANGE - i) as u64)
            };

            let output_field = if quantized >= 0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            table.add_entry(vec![input_field], vec![output_field]);
        }

        table.set_error_bound(F::from((max_error * scale_f).ceil() as u64 + 1));
        (table, max_error)
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Returns the maximum error compared to exact GELU.
    pub fn max_error(&self) -> f64 {
        self.max_error
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for FastGELULookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// GELU Derivative Lookup (for training)
// ---------------------------------------------------------------------------

/// GELU derivative lookup for backpropagation.
///
/// d/dx GELU(x) = Φ(x) + x * φ(x)
///
/// where Φ is the CDF and φ is the PDF of the standard normal.
#[derive(Clone)]
pub struct GELUDerivativeLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> GELUDerivativeLookup<F, RANGE, SCALE> {
    /// Creates a new GELU derivative lookup table.
    pub fn new() -> Self {
        Self {
            table: Self::build_table(),
            _marker: PhantomData,
        }
    }

    fn build_table() -> PlookupTable<F> {
        let entries = gelu_derivative_entries::<F>(RANGE, SCALE);
        let mut table = PlookupTable::new(SCALE);
        for (input, output) in entries {
            table.add_entry(vec![input], vec![output]);
        }
        table.set_error_bound(F::from(2)); // Derivative approximation error
        table
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Computes GELU derivative.
    pub fn compute(x: f64) -> f64 {
        gelu_derivative(x)
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for GELUDerivativeLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the GELU derivative.
fn gelu_derivative(x: f64) -> f64 {
    let cdf = 0.5 * (1.0 + (SQRT_2_PI * (x + GELU_CUBIC_COEFF * x.powi(3))).tanh());
    let pdf = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp();
    cdf + x * pdf
}

/// Generates GELU derivative table entries.
pub fn gelu_derivative_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let half = range / 2;
    let scale_f = scale as f64;
    let mut entries = Vec::with_capacity(range);

    for i in 0..range {
        let x = if i < half {
            i as f64 / scale_f
        } else {
            -((range - i) as f64) / scale_f
        };

        let deriv = gelu_derivative(x);
        let quantized = (deriv * scale_f).round() as i64;

        let input_field = if i < half {
            F::from(i as u64)
        } else {
            F::ZERO - F::from((range - i) as u64)
        };

        let output_field = if quantized >= 0 {
            F::from(quantized as u64)
        } else {
            F::ZERO - F::from((-quantized) as u64)
        };

        entries.push((input_field, output_field));
    }

    entries
}

// ---------------------------------------------------------------------------
// SiLU (Swish) Lookup
// ---------------------------------------------------------------------------

/// SiLU (Swish) lookup table.
///
/// SiLU(x) = x * sigmoid(x)
///
/// Also known as Swish activation, used in EfficientNet and other modern architectures.
#[derive(Clone)]
pub struct SiLULookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> SiLULookup<F, RANGE, SCALE> {
    /// Creates a new SiLU lookup table.
    pub fn new() -> Self {
        Self {
            table: Self::build_table(),
            _marker: PhantomData,
        }
    }

    fn build_table() -> PlookupTable<F> {
        let half = RANGE / 2;
        let scale_f = SCALE as f64;
        let mut table = PlookupTable::new(SCALE);
        let mut max_error = 0.0f64;

        for i in 0..RANGE {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            let silu = x * sigmoid(x);
            let quantized = (silu * scale_f).round() as i64;
            let error = (silu - quantized as f64 / scale_f).abs();
            max_error = max_error.max(error);

            let input_field = if i < half {
                F::from(i as u64)
            } else {
                F::ZERO - F::from((RANGE - i) as u64)
            };

            let output_field = if quantized >= 0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            table.add_entry(vec![input_field], vec![output_field]);
        }

        table.set_error_bound(F::from((max_error * scale_f).ceil() as u64 + 1));
        table
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Computes SiLU.
    pub fn compute(x: f64) -> f64 {
        x * sigmoid(x)
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for SiLULookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::Circuit,
    };
    use halo2curves::bn256::Fr;
    use halo2curves::ff::Field;

    #[test]
    fn test_gelu_exact() {
        // GELU(0) = 0
        assert!((gelu_exact(0.0) - 0.0).abs() < 1e-6);

        // GELU(x) ≈ x for large positive x
        assert!((gelu_exact(5.0) - 5.0).abs() < 0.01);

        // GELU(x) ≈ 0 for large negative x
        assert!(gelu_exact(-5.0).abs() < 0.01);

        // GELU is roughly linear around 0
        let g1 = gelu_exact(0.5);
        assert!(g1 > 0.0 && g1 < 0.5);
    }

    #[test]
    fn test_gelu_table_entries() {
        let entries = gelu_table_entries::<Fr>(256, 64);
        assert_eq!(entries.len(), 256);

        // Check that GELU(0) ≈ 0
        // Find the entry for input 0
        let zero_entry = entries.iter().find(|(i, _)| *i == Fr::ZERO);
        assert!(zero_entry.is_some());
        // Output should be small (close to 0)
    }

    #[test]
    fn test_gelu_lookup() {
        let lookup = GELULookup::<Fr, 256, 64>::new();

        // Check table size
        assert_eq!(lookup.table().len(), 256);

        // Check error bound is reasonable for INT8 quantization
        // With 256 entries and scale=64, max_error is bounded by quantization step
        assert!(lookup.max_error() < 0.5, "max_error = {}", lookup.max_error());
    }

    #[test]
    fn test_fast_gelu_lookup() {
        let lookup = FastGELULookup::<Fr, 256, 64>::new();

        // FastGELU should have small error compared to exact GELU
        assert!(lookup.max_error() < 0.05);
    }

    #[test]
    fn test_gelu_derivative() {
        // d/dx GELU(0) ≈ 0.5
        let d0 = gelu_derivative(0.0);
        assert!((d0 - 0.5).abs() < 0.01);

        // d/dx GELU(x) → 1 for large positive x
        let d_pos = gelu_derivative(5.0);
        assert!((d_pos - 1.0).abs() < 0.01);

        // d/dx GELU(x) → 0 for large negative x
        let d_neg = gelu_derivative(-5.0);
        assert!(d_neg.abs() < 0.01);
    }

    #[test]
    fn test_silu_lookup() {
        let lookup = SiLULookup::<Fr, 256, 64>::new();

        // SiLU(0) = 0
        let zero_result = lookup.table().lookup(&[Fr::ZERO]);
        assert!(zero_result.is_some());
        // Output should be close to 0
    }

    // Integration test with actual circuit
    #[derive(Clone)]
    struct GELUTestCircuit {
        input: Fr,
        expected_output: Fr,
    }

    impl Default for GELUTestCircuit {
        fn default() -> Self {
            Self {
                input: Fr::ZERO,
                expected_output: Fr::ZERO,
            }
        }
    }

    impl Circuit<Fr> for GELUTestCircuit {
        type Config = GELULookupConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice_in = meta.advice_column();
            let advice_out = meta.advice_column();
            meta.enable_equality(advice_in);
            meta.enable_equality(advice_out);
            GELUChip::<Fr, 256, 64>::configure(meta, advice_in, advice_out)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = GELUChip::<Fr, 256, 64>::new(config);
            chip.load_padded(&mut layouter, 1024)?;

            layouter.assign_region(
                || "gelu_test",
                |mut region| {
                    chip.gelu(
                        &mut region,
                        0,
                        Value::known(self.input),
                        Value::known(self.expected_output),
                    )
                },
            )
        }
    }

    #[test]
    fn test_gelu_circuit_zero() {
        // GELU(0) = 0
        let circuit = GELUTestCircuit {
            input: Fr::ZERO,
            expected_output: Fr::ZERO,
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_gelu_circuit_positive() {
        // GELU(64) with scale=64 means GELU(1.0)
        // GELU(1.0) ≈ 0.8413, so scaled output ≈ 54
        let input = Fr::from(64); // x = 1.0 scaled
        let expected = Fr::from(54); // GELU(1.0) * 64 ≈ 54

        let circuit = GELUTestCircuit {
            input,
            expected_output: expected,
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
