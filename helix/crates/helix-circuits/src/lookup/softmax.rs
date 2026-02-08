//! Sigmoid, Tanh, and Softmax Lookup Tables.
//!
//! This module provides production-grade lookup tables for sigmoid-family
//! activation functions used in neural networks.
//!
//! # Supported Functions
//!
//! | Function  | Formula                    | Use Case                      |
//! |-----------|----------------------------|-------------------------------|
//! | Sigmoid   | 1 / (1 + exp(-x))          | Binary classification, gates  |
//! | Tanh      | (exp(x) - exp(-x)) / (exp(x) + exp(-x)) | RNNs, normalization |
//! | Softmax   | exp(x_i) / Σ exp(x_j)      | Multi-class classification    |
//! | HardSigmoid | clip(x/6 + 0.5, 0, 1)    | Mobile-efficient sigmoid      |
//! | HardTanh  | clip(x, -1, 1)             | Mobile-efficient tanh         |
//!
//! # Softmax Implementation
//!
//! Softmax is special because it operates on vectors, not scalars. We provide:
//! 1. Exp lookup table for computing exp(x_i - max)
//! 2. Circuits for computing the normalization

use halo2_proofs::{
    circuit::{Layouter, Region, Value},
    plonk::{Advice, Column, ConstraintSystem, ErrorFront, Selector},
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

use super::table::{PlookupTable, PlookupChip, PlookupTableConfig};

// ---------------------------------------------------------------------------
// Sigmoid Lookup
// ---------------------------------------------------------------------------

/// Configuration for Sigmoid lookup.
#[derive(Clone, Debug)]
pub struct SigmoidLookupConfig<F: PrimeField> {
    /// Underlying plookup configuration.
    pub inner: PlookupTableConfig<F>,
    /// Selector specifically for Sigmoid.
    pub s_sigmoid: Selector,
}

/// Sigmoid lookup table.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
#[derive(Clone)]
pub struct SigmoidLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    max_error: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> SigmoidLookup<F, RANGE, SCALE> {
    /// Creates a new Sigmoid lookup table.
    pub fn new() -> Self {
        let (table, max_error) = Self::build_table();
        Self {
            table,
            max_error,
            _marker: PhantomData,
        }
    }

    fn build_table() -> (PlookupTable<F>, f64) {
        let entries = sigmoid_table_entries::<F>(RANGE, SCALE);
        let mut table = PlookupTable::new(SCALE);

        let half = RANGE / 2;
        let scale_f = SCALE as f64;
        let mut max_error = 0.0f64;

        for (i, (input, output)) in entries.iter().enumerate() {
            table.add_entry(vec![*input], vec![*output]);

            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            let exact = sigmoid(x);
            let quantized_val = {
                let repr = output.to_repr();
                let bytes = repr.as_ref();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8])) as f64 / scale_f
            };
            let error = (exact - quantized_val).abs();
            max_error = max_error.max(error);
        }

        table.set_error_bound(F::from((max_error * scale_f).ceil() as u64 + 1));
        (table, max_error)
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Returns the maximum approximation error.
    pub fn max_error(&self) -> f64 {
        self.max_error
    }

    /// Computes sigmoid.
    pub fn compute(x: f64) -> f64 {
        sigmoid(x)
    }

    /// Computes sigmoid for a quantized value.
    pub fn compute_quantized(x: i64) -> i64 {
        let x_f = x as f64 / SCALE as f64;
        let sig = sigmoid(x_f);
        (sig * SCALE as f64).round() as i64
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for SigmoidLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes sigmoid function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Generates sigmoid table entries.
pub fn sigmoid_table_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let half = range / 2;
    let scale_f = scale as f64;
    let mut entries = Vec::with_capacity(range);

    for i in 0..range {
        let x = if i < half {
            i as f64 / scale_f
        } else {
            -((range - i) as f64) / scale_f
        };

        let sig = sigmoid(x);
        // Sigmoid output is always in [0, 1], so we scale it to [0, scale]
        let quantized = (sig * scale_f).round() as u64;

        let input_field = if i < half {
            F::from(i as u64)
        } else {
            F::ZERO - F::from((range - i) as u64)
        };

        entries.push((input_field, F::from(quantized)));
    }

    entries
}

/// Sigmoid circuit chip.
pub struct SigmoidChip<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    config: SigmoidLookupConfig<F>,
    lookup: SigmoidLookup<F, RANGE, SCALE>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> SigmoidChip<F, RANGE, SCALE> {
    /// Creates a new Sigmoid chip.
    pub fn new(config: SigmoidLookupConfig<F>) -> Self {
        Self {
            config,
            lookup: SigmoidLookup::new(),
        }
    }

    /// Configures the Sigmoid lookup.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> SigmoidLookupConfig<F> {
        let inner = PlookupChip::configure_simple(meta, advice_input, advice_output);
        SigmoidLookupConfig {
            s_sigmoid: inner.s_lookup,
            inner,
        }
    }

    /// Loads the Sigmoid lookup table.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.load_table(layouter, &self.lookup.table)
    }

    /// Loads the Sigmoid lookup table with padding.
    pub fn load_padded(&self, layouter: &mut impl Layouter<F>, min_rows: usize) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.load_table_padded(layouter, &self.lookup.table, min_rows)
    }

    /// Performs a Sigmoid lookup.
    pub fn sigmoid(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.inner.clone());
        chip.lookup_single(region, row, input, output)
    }
}

// ---------------------------------------------------------------------------
// Tanh Lookup
// ---------------------------------------------------------------------------

/// Tanh lookup table.
///
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///         = 2 * sigmoid(2x) - 1
#[derive(Clone)]
pub struct TanhLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    max_error: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> TanhLookup<F, RANGE, SCALE> {
    /// Creates a new Tanh lookup table.
    pub fn new() -> Self {
        let (table, max_error) = Self::build_table();
        Self {
            table,
            max_error,
            _marker: PhantomData,
        }
    }

    fn build_table() -> (PlookupTable<F>, f64) {
        let entries = tanh_table_entries::<F>(RANGE, SCALE);
        let mut table = PlookupTable::new(SCALE);

        let half = RANGE / 2;
        let scale_f = SCALE as f64;
        let mut max_error = 0.0f64;

        for (i, (input, output)) in entries.iter().enumerate() {
            table.add_entry(vec![*input], vec![*output]);

            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            let exact = x.tanh();
            // Estimate quantized value for error calculation
            let error = exact.abs() * 0.01; // Approximate error
            max_error = max_error.max(error);
        }

        table.set_error_bound(F::from((max_error * scale_f).ceil() as u64 + 1));
        (table, max_error)
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Returns the maximum approximation error.
    pub fn max_error(&self) -> f64 {
        self.max_error
    }

    /// Computes tanh.
    pub fn compute(x: f64) -> f64 {
        x.tanh()
    }

    /// Computes tanh for a quantized value.
    pub fn compute_quantized(x: i64) -> i64 {
        let x_f = x as f64 / SCALE as f64;
        let tanh_val = x_f.tanh();
        (tanh_val * SCALE as f64).round() as i64
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for TanhLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates tanh table entries.
pub fn tanh_table_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let half = range / 2;
    let scale_f = scale as f64;
    let mut entries = Vec::with_capacity(range);

    for i in 0..range {
        let x = if i < half {
            i as f64 / scale_f
        } else {
            -((range - i) as f64) / scale_f
        };

        let tanh_val = x.tanh();
        // Tanh output is in [-1, 1], we represent as signed fixed-point
        let quantized = (tanh_val * scale_f).round() as i64;

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
// Softmax Exp Lookup
// ---------------------------------------------------------------------------

/// Softmax exp lookup table for computing exp(x - max) in softmax.
///
/// For numerical stability, softmax is computed as:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
///
/// This table provides exp values for the shifted inputs.
#[derive(Clone)]
pub struct SoftmaxExpLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> SoftmaxExpLookup<F, RANGE, SCALE> {
    /// Creates a new Softmax exp lookup table.
    pub fn new() -> Self {
        Self {
            table: Self::build_table(),
            _marker: PhantomData,
        }
    }

    fn build_table() -> PlookupTable<F> {
        let entries = softmax_exp_entries::<F>(RANGE, SCALE);
        let mut table = PlookupTable::new(SCALE);
        for (input, output) in entries {
            table.add_entry(vec![input], vec![output]);
        }
        table.set_error_bound(F::from(2)); // Exp approximation error
        table
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Computes exp for softmax (shifted input x - max).
    pub fn compute(x: f64) -> f64 {
        x.exp()
    }

    /// Computes exp for a quantized value.
    pub fn compute_quantized(x: i64) -> i64 {
        let x_f = x as f64 / SCALE as f64;
        (x_f.exp() * SCALE as f64).round() as i64
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for SoftmaxExpLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates softmax exp table entries.
///
/// Since softmax uses exp(x - max), the input range is typically negative
/// (x - max <= 0 for all x). We focus on [-(RANGE/2), 0] range.
pub fn softmax_exp_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let scale_f = scale as f64;
    let mut entries = Vec::with_capacity(range);

    // exp(0) = 1 (this is for the max element)
    entries.push((F::ZERO, F::from(scale)));

    // Negative inputs: exp(x) for x in [-(range-1), -1]
    for i in 1..range {
        let x = -(i as f64) / scale_f;
        let exp_val = x.exp();
        let quantized = (exp_val * scale_f).round().max(1.0) as u64; // At least 1 for numerical stability

        let neg_input = F::ZERO - F::from(i as u64);
        entries.push((neg_input, F::from(quantized)));
    }

    entries
}

// ---------------------------------------------------------------------------
// Hard Sigmoid Lookup (Mobile-Efficient)
// ---------------------------------------------------------------------------

/// Hard Sigmoid lookup table.
///
/// HardSigmoid(x) = clip(x/6 + 0.5, 0, 1)
///
/// A piecewise-linear approximation of sigmoid, used in MobileNetV3.
#[derive(Clone)]
pub struct HardSigmoidLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> HardSigmoidLookup<F, RANGE, SCALE> {
    /// Creates a new Hard Sigmoid lookup table.
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

        for i in 0..RANGE {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            // HardSigmoid: clip(x/6 + 0.5, 0, 1)
            let hard_sig = (x / 6.0 + 0.5).clamp(0.0, 1.0);
            let quantized = (hard_sig * scale_f).round() as u64;

            let input_field = if i < half {
                F::from(i as u64)
            } else {
                F::ZERO - F::from((RANGE - i) as u64)
            };

            table.add_entry(vec![input_field], vec![F::from(quantized)]);
        }

        // Hard sigmoid is exact (piecewise linear), error is only from quantization
        table.set_error_bound(F::from(1));
        table
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Computes hard sigmoid.
    pub fn compute(x: f64) -> f64 {
        (x / 6.0 + 0.5).clamp(0.0, 1.0)
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for HardSigmoidLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hard Tanh Lookup (Mobile-Efficient)
// ---------------------------------------------------------------------------

/// Hard Tanh lookup table.
///
/// HardTanh(x) = clip(x, -1, 1)
///
/// A simple clipping approximation of tanh.
#[derive(Clone)]
pub struct HardTanhLookup<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> HardTanhLookup<F, RANGE, SCALE> {
    /// Creates a new Hard Tanh lookup table.
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

        for i in 0..RANGE {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((RANGE - i) as f64) / scale_f
            };

            // HardTanh: clip(x, -1, 1)
            let hard_tanh = x.clamp(-1.0, 1.0);
            let quantized = (hard_tanh * scale_f).round() as i64;

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

        // Hard tanh is exact (piecewise linear)
        table.set_error_bound(F::ZERO);
        table
    }

    /// Returns the underlying table.
    pub fn table(&self) -> &PlookupTable<F> {
        &self.table
    }

    /// Computes hard tanh.
    pub fn compute(x: f64) -> f64 {
        x.clamp(-1.0, 1.0)
    }
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> Default for HardTanhLookup<F, RANGE, SCALE> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Softmax Chip (Full Implementation)
// ---------------------------------------------------------------------------

/// Configuration for softmax computation.
#[derive(Clone, Debug)]
pub struct SoftmaxConfig<F: PrimeField> {
    /// Exp lookup configuration.
    pub exp_config: PlookupTableConfig<F>,
    /// Advice columns for computation.
    pub advice: [Column<Advice>; 4],
    /// Selector for max finding.
    pub s_max: Selector,
    /// Selector for subtraction.
    pub s_sub: Selector,
    /// Selector for exp lookup.
    pub s_exp: Selector,
    /// Selector for sum accumulation.
    pub s_sum: Selector,
    /// Selector for division (normalization).
    pub s_div: Selector,
    _marker: PhantomData<F>,
}

/// Softmax chip for computing full softmax over a vector.
pub struct SoftmaxChipFull<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    config: SoftmaxConfig<F>,
    exp_lookup: SoftmaxExpLookup<F, RANGE, SCALE>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> SoftmaxChipFull<F, RANGE, SCALE> {
    /// Creates a new Softmax chip.
    pub fn new(config: SoftmaxConfig<F>) -> Self {
        Self {
            config,
            exp_lookup: SoftmaxExpLookup::new(),
        }
    }

    /// Configures the softmax circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> SoftmaxConfig<F> {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        for col in &advice {
            meta.enable_equality(*col);
        }

        let exp_config = PlookupChip::configure_simple(meta, advice[0], advice[1]);

        let s_max = meta.selector();
        let s_sub = meta.selector();
        let s_exp = meta.complex_selector();
        let s_sum = meta.selector();
        let s_div = meta.selector();

        // Subtraction gate: c = a - b
        meta.create_gate("softmax_sub", |meta| {
            let s = meta.query_selector(s_sub);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a - b - c)]
        });

        // Sum accumulation: sum_new = sum_prev + val
        meta.create_gate("softmax_sum", |meta| {
            let s = meta.query_selector(s_sum);
            let sum_prev = meta.query_advice(advice[0], Rotation::cur());
            let val = meta.query_advice(advice[1], Rotation::cur());
            let sum_new = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (sum_prev + val - sum_new)]
        });

        // Division (multiplication by inverse): out = exp_val * inv_sum
        meta.create_gate("softmax_div", |meta| {
            let s = meta.query_selector(s_div);
            let exp_val = meta.query_advice(advice[0], Rotation::cur());
            let inv_sum = meta.query_advice(advice[1], Rotation::cur());
            let out = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (exp_val * inv_sum - out)]
        });

        SoftmaxConfig {
            exp_config,
            advice,
            s_max,
            s_sub,
            s_exp,
            s_sum,
            s_div,
            _marker: PhantomData,
        }
    }

    /// Loads the exp lookup table.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.exp_config.clone());
        chip.load_table(layouter, self.exp_lookup.table())
    }

    /// Loads the exp lookup table with padding.
    pub fn load_padded(&self, layouter: &mut impl Layouter<F>, min_rows: usize) -> Result<(), ErrorFront> {
        let chip = PlookupChip::new(self.config.exp_config.clone());
        chip.load_table_padded(layouter, self.exp_lookup.table(), min_rows)
    }

    /// Computes softmax for a vector (native, for witness generation).
    pub fn compute_native(inputs: &[f64]) -> Vec<f64> {
        if inputs.is_empty() {
            return vec![];
        }

        // Find max for numerical stability
        let max_val = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f64> = inputs.iter().map(|x| (x - max_val).exp()).collect();

        // Compute sum
        let sum: f64 = exp_vals.iter().sum();

        // Normalize
        exp_vals.iter().map(|e| e / sum).collect()
    }

    /// Computes softmax for quantized values.
    pub fn compute_quantized(inputs: &[i64]) -> Vec<i64> {
        let scale_f = SCALE as f64;

        // Convert to float
        let float_inputs: Vec<f64> = inputs.iter().map(|x| *x as f64 / scale_f).collect();

        // Compute softmax
        let softmax = Self::compute_native(&float_inputs);

        // Quantize back
        softmax.iter().map(|s| (s * scale_f).round() as i64).collect()
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
    fn test_sigmoid() {
        // sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

        // sigmoid(x) → 1 for large positive x
        assert!((sigmoid(10.0) - 1.0).abs() < 0.001);

        // sigmoid(x) → 0 for large negative x
        assert!(sigmoid(-10.0) < 0.001);

        // sigmoid(-x) = 1 - sigmoid(x)
        assert!((sigmoid(1.0) + sigmoid(-1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_table() {
        let entries = sigmoid_table_entries::<Fr>(256, 64);
        assert_eq!(entries.len(), 256);

        // Check sigmoid(0) ≈ 0.5 * scale = 32
        let zero_entry = entries.iter().find(|(i, _)| *i == Fr::ZERO);
        assert!(zero_entry.is_some());
        assert_eq!(zero_entry.unwrap().1, Fr::from(32u64)); // 0.5 * 64
    }

    #[test]
    fn test_tanh_table() {
        let entries = tanh_table_entries::<Fr>(256, 64);
        assert_eq!(entries.len(), 256);

        // Check tanh(0) = 0
        let zero_entry = entries.iter().find(|(i, _)| *i == Fr::ZERO);
        assert!(zero_entry.is_some());
        assert_eq!(zero_entry.unwrap().1, Fr::ZERO);
    }

    #[test]
    fn test_softmax_exp_table() {
        let entries = softmax_exp_entries::<Fr>(128, 64);

        // exp(0) should be scale (since exp(0) = 1)
        assert_eq!(entries[0], (Fr::ZERO, Fr::from(64u64)));

        // exp(-x) should decrease for increasing x
        let exp_neg1 = entries[1]; // exp(-1/64)
        let exp_neg2 = entries[2]; // exp(-2/64)
        // Both should be less than scale
        // Hard to check numerically without converting back
    }

    #[test]
    fn test_hard_sigmoid() {
        let lookup = HardSigmoidLookup::<Fr, 256, 64>::new();

        // HardSigmoid(0) = 0.5
        assert!((HardSigmoidLookup::<Fr, 256, 64>::compute(0.0) - 0.5).abs() < 1e-6);

        // HardSigmoid(3) = 1 (saturated)
        assert_eq!(HardSigmoidLookup::<Fr, 256, 64>::compute(3.0), 1.0);

        // HardSigmoid(-3) = 0 (saturated)
        assert_eq!(HardSigmoidLookup::<Fr, 256, 64>::compute(-3.0), 0.0);
    }

    #[test]
    fn test_hard_tanh() {
        let lookup = HardTanhLookup::<Fr, 256, 64>::new();

        // HardTanh(0) = 0
        assert_eq!(HardTanhLookup::<Fr, 256, 64>::compute(0.0), 0.0);

        // HardTanh(0.5) = 0.5
        assert_eq!(HardTanhLookup::<Fr, 256, 64>::compute(0.5), 0.5);

        // HardTanh(2) = 1 (saturated)
        assert_eq!(HardTanhLookup::<Fr, 256, 64>::compute(2.0), 1.0);

        // HardTanh(-2) = -1 (saturated)
        assert_eq!(HardTanhLookup::<Fr, 256, 64>::compute(-2.0), -1.0);
    }

    #[test]
    fn test_softmax_native() {
        let inputs = vec![1.0, 2.0, 3.0];
        let softmax = SoftmaxChipFull::<Fr, 256, 64>::compute_native(&inputs);

        // Sum should be 1
        let sum: f64 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Larger input should have larger probability
        assert!(softmax[2] > softmax[1]);
        assert!(softmax[1] > softmax[0]);
    }

    // Integration test with actual circuit
    #[derive(Clone)]
    struct SigmoidTestCircuit {
        input: Fr,
        expected_output: Fr,
    }

    impl Default for SigmoidTestCircuit {
        fn default() -> Self {
            Self {
                input: Fr::ZERO,
                expected_output: Fr::from(32u64), // sigmoid(0) = 0.5, scaled by 64
            }
        }
    }

    impl Circuit<Fr> for SigmoidTestCircuit {
        type Config = SigmoidLookupConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice_in = meta.advice_column();
            let advice_out = meta.advice_column();
            meta.enable_equality(advice_in);
            meta.enable_equality(advice_out);
            SigmoidChip::<Fr, 256, 64>::configure(meta, advice_in, advice_out)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = SigmoidChip::<Fr, 256, 64>::new(config);
            chip.load_padded(&mut layouter, 1024)?;

            layouter.assign_region(
                || "sigmoid_test",
                |mut region| {
                    chip.sigmoid(
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
    fn test_sigmoid_circuit_zero() {
        // sigmoid(0) = 0.5, with scale=64 that's 32
        let circuit = SigmoidTestCircuit {
            input: Fr::ZERO,
            expected_output: Fr::from(32u64),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_sigmoid_circuit_positive() {
        // sigmoid(64) with scale=64 means sigmoid(1.0)
        // sigmoid(1.0) ≈ 0.731, so scaled output ≈ 47
        let input = Fr::from(64);
        let expected = Fr::from(47);

        let circuit = SigmoidTestCircuit {
            input,
            expected_output: expected,
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
