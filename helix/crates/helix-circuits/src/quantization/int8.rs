//! INT8 Quantization Circuits.
//!
//! This module provides comprehensive INT8 quantization circuits with support for:
//! - Symmetric quantization (zero-point = 0)
//! - Asymmetric quantization (arbitrary zero-point)
//! - Per-tensor and per-channel scales
//! - Quantized matrix multiplication verification
//!
//! # Quantization Formula
//!
//! For symmetric quantization:
//!   q = round(x / scale)
//!   x_approx = q * scale
//!
//! For asymmetric quantization:
//!   q = round(x / scale) + zero_point
//!   x_approx = (q - zero_point) * scale
//!
//! # Error Bounds
//!
//! Per-value quantization error: |x - x_approx| <= 0.5 * scale
//! This bound is tight and achieved when x is exactly between two quantization levels.

use halo2_proofs::{
    circuit::{Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, ErrorFront, Expression,
        Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

use super::{INT8_MAX, INT8_MIN, INT8_RANGE};

/// Configuration for INT8 quantization circuits.
#[derive(Clone, Debug)]
pub struct Int8QuantConfig<F: PrimeField> {
    /// Advice columns for values.
    pub value: Column<Advice>,
    pub quantized: Column<Advice>,
    pub scale: Column<Advice>,
    pub zero_point: Column<Advice>,
    pub error: Column<Advice>,

    /// Additional advice for operations.
    pub aux: [Column<Advice>; 3],

    /// Lookup table for INT8 range check.
    pub int8_table: TableColumn,

    /// Selectors.
    pub s_quantize: Selector,
    pub s_dequantize: Selector,
    pub s_int8_range: Selector,
    pub s_mul: Selector,
    pub s_add: Selector,
    pub s_requantize: Selector,

    _marker: PhantomData<F>,
}

/// Symmetric INT8 quantization parameters.
#[derive(Clone, Debug)]
pub struct Int8SymmetricParams {
    /// Scale factor (real_value = quantized_value * scale).
    pub scale: f64,
    /// Computed from data range.
    pub data_min: f64,
    pub data_max: f64,
}

impl Int8SymmetricParams {
    /// Creates parameters from data range.
    pub fn from_range(data_min: f64, data_max: f64) -> Self {
        let abs_max = data_min.abs().max(data_max.abs());
        let scale = abs_max / INT8_MAX as f64;
        Self {
            scale,
            data_min,
            data_max,
        }
    }

    /// Quantizes a floating-point value.
    pub fn quantize(&self, x: f64) -> i8 {
        let q = (x / self.scale).round();
        q.clamp(INT8_MIN as f64, INT8_MAX as f64) as i8
    }

    /// Dequantizes an INT8 value.
    pub fn dequantize(&self, q: i8) -> f64 {
        q as f64 * self.scale
    }

    /// Returns the maximum quantization error.
    pub fn max_error(&self) -> f64 {
        0.5 * self.scale
    }
}

/// Asymmetric INT8 quantization parameters.
#[derive(Clone, Debug)]
pub struct Int8AsymmetricParams {
    /// Scale factor.
    pub scale: f64,
    /// Zero-point offset.
    pub zero_point: i32,
    /// Data range.
    pub data_min: f64,
    pub data_max: f64,
}

impl Int8AsymmetricParams {
    /// Creates parameters from data range.
    pub fn from_range(data_min: f64, data_max: f64) -> Self {
        let scale = (data_max - data_min) / (INT8_MAX - INT8_MIN) as f64;
        let zero_point = INT8_MIN as f64 - data_min / scale;
        Self {
            scale,
            zero_point: zero_point.round() as i32,
            data_min,
            data_max,
        }
    }

    /// Quantizes a floating-point value.
    pub fn quantize(&self, x: f64) -> i8 {
        let q = (x / self.scale).round() + self.zero_point as f64;
        q.clamp(INT8_MIN as f64, INT8_MAX as f64) as i8
    }

    /// Dequantizes an INT8 value.
    pub fn dequantize(&self, q: i8) -> f64 {
        (q as f64 - self.zero_point as f64) * self.scale
    }

    /// Returns the maximum quantization error.
    pub fn max_error(&self) -> f64 {
        0.5 * self.scale
    }
}

/// Per-channel INT8 quantization parameters.
#[derive(Clone, Debug)]
pub struct PerChannelInt8Params {
    /// Per-channel scales.
    pub scales: Vec<f64>,
    /// Per-channel zero-points (for asymmetric).
    pub zero_points: Vec<i32>,
    /// Number of channels.
    pub num_channels: usize,
    /// Whether using symmetric quantization.
    pub symmetric: bool,
}

impl PerChannelInt8Params {
    /// Creates symmetric per-channel params from channel ranges.
    pub fn symmetric_from_ranges(ranges: &[(f64, f64)]) -> Self {
        let scales: Vec<f64> = ranges
            .iter()
            .map(|(min, max)| {
                let abs_max = min.abs().max(max.abs());
                abs_max / INT8_MAX as f64
            })
            .collect();

        let num_channels = ranges.len();
        Self {
            scales,
            zero_points: vec![0; num_channels],
            num_channels,
            symmetric: true,
        }
    }

    /// Creates asymmetric per-channel params from channel ranges.
    pub fn asymmetric_from_ranges(ranges: &[(f64, f64)]) -> Self {
        let mut scales = Vec::with_capacity(ranges.len());
        let mut zero_points = Vec::with_capacity(ranges.len());

        for (min, max) in ranges {
            let scale = (max - min) / (INT8_MAX - INT8_MIN) as f64;
            let zp = INT8_MIN as f64 - min / scale;
            scales.push(scale);
            zero_points.push(zp.round() as i32);
        }

        Self {
            scales,
            zero_points,
            num_channels: ranges.len(),
            symmetric: false,
        }
    }

    /// Quantizes a value in a specific channel.
    pub fn quantize(&self, x: f64, channel: usize) -> i8 {
        let q = (x / self.scales[channel]).round() + self.zero_points[channel] as f64;
        q.clamp(INT8_MIN as f64, INT8_MAX as f64) as i8
    }

    /// Dequantizes an INT8 value from a specific channel.
    pub fn dequantize(&self, q: i8, channel: usize) -> f64 {
        (q as f64 - self.zero_points[channel] as f64) * self.scales[channel]
    }
}

/// Witness for INT8 quantization.
#[derive(Clone, Debug)]
pub struct Int8QuantWitness<F: PrimeField> {
    /// Original floating-point value (scaled to fixed-point).
    pub original: F,
    /// Quantized INT8 value.
    pub quantized: F,
    /// Scale factor (as fixed-point).
    pub scale: F,
    /// Zero-point (for asymmetric).
    pub zero_point: F,
    /// Quantization error.
    pub error: F,
}

impl<F: PrimeField> Int8QuantWitness<F> {
    /// Creates a witness for symmetric quantization.
    pub fn symmetric(original: f64, params: &Int8SymmetricParams, fp_scale: u64) -> Self {
        let q = params.quantize(original);
        let dequant = params.dequantize(q);
        let error = original - dequant;

        // Convert to field elements with fixed-point scaling
        let original_fp = (original * fp_scale as f64).round() as i64;
        let error_fp = (error * fp_scale as f64).round() as i64;
        let scale_fp = (params.scale * fp_scale as f64).round() as u64;

        Self {
            original: Self::i64_to_field(original_fp),
            quantized: Self::i64_to_field(q as i64),
            scale: F::from(scale_fp),
            zero_point: F::ZERO,
            error: Self::i64_to_field(error_fp),
        }
    }

    /// Creates a witness for asymmetric quantization.
    pub fn asymmetric(original: f64, params: &Int8AsymmetricParams, fp_scale: u64) -> Self {
        let q = params.quantize(original);
        let dequant = params.dequantize(q);
        let error = original - dequant;

        let original_fp = (original * fp_scale as f64).round() as i64;
        let error_fp = (error * fp_scale as f64).round() as i64;
        let scale_fp = (params.scale * fp_scale as f64).round() as u64;

        Self {
            original: Self::i64_to_field(original_fp),
            quantized: Self::i64_to_field(q as i64),
            scale: F::from(scale_fp),
            zero_point: Self::i64_to_field(params.zero_point as i64),
            error: Self::i64_to_field(error_fp),
        }
    }

    fn i64_to_field(x: i64) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

/// INT8 quantization chip.
pub struct Int8QuantChip<F: PrimeField> {
    config: Int8QuantConfig<F>,
}

impl<F: PrimeField> Int8QuantChip<F> {
    /// Creates a new INT8 quantization chip.
    pub fn new(config: Int8QuantConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the INT8 quantization circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Int8QuantConfig<F> {
        // Advice columns
        let value = meta.advice_column();
        let quantized = meta.advice_column();
        let scale = meta.advice_column();
        let zero_point = meta.advice_column();
        let error = meta.advice_column();
        let aux = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // Enable equality for all advice columns
        for col in [value, quantized, scale, zero_point, error] {
            meta.enable_equality(col);
        }
        for col in &aux {
            meta.enable_equality(*col);
        }

        // Lookup table for INT8 range
        let int8_table = meta.lookup_table_column();

        // Selectors
        let s_quantize = meta.selector();
        let s_dequantize = meta.selector();
        let s_int8_range = meta.complex_selector();
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_requantize = meta.selector();

        // INT8 range check via lookup
        // The quantized value (shifted to unsigned) must be in [0, 255]
        meta.lookup("int8_range_check", |meta| {
            let s = meta.query_selector(s_int8_range);
            let q = meta.query_advice(quantized, Rotation::cur());
            // Shift signed to unsigned: q + 128
            let shifted = q + Expression::Constant(F::from(128u64));
            vec![(s * shifted, int8_table)]
        });

        // Quantization constraint:
        // For symmetric: original = quantized * scale + error
        // For asymmetric: original = (quantized - zero_point) * scale + error
        meta.create_gate("quantize", |meta| {
            let s = meta.query_selector(s_quantize);
            let orig = meta.query_advice(value, Rotation::cur());
            let q = meta.query_advice(quantized, Rotation::cur());
            let scl = meta.query_advice(scale, Rotation::cur());
            let zp = meta.query_advice(zero_point, Rotation::cur());
            let err = meta.query_advice(error, Rotation::cur());

            // original = (quantized - zero_point) * scale + error
            vec![s * (orig - (q.clone() - zp) * scl - err)]
        });

        // Dequantization constraint:
        // value = (quantized - zero_point) * scale
        meta.create_gate("dequantize", |meta| {
            let s = meta.query_selector(s_dequantize);
            let val = meta.query_advice(value, Rotation::cur());
            let q = meta.query_advice(quantized, Rotation::cur());
            let scl = meta.query_advice(scale, Rotation::cur());
            let zp = meta.query_advice(zero_point, Rotation::cur());

            vec![s * (val - (q - zp) * scl)]
        });

        // Multiplication: c = a * b
        meta.create_gate("int8_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(aux[0], Rotation::cur());
            let b = meta.query_advice(aux[1], Rotation::cur());
            let c = meta.query_advice(aux[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        // Addition: c = a + b
        meta.create_gate("int8_add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(aux[0], Rotation::cur());
            let b = meta.query_advice(aux[1], Rotation::cur());
            let c = meta.query_advice(aux[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        // Requantization: output = round(accumulator / output_scale)
        // Verified as: accumulator = output * output_scale + error
        meta.create_gate("requantize", |meta| {
            let s = meta.query_selector(s_requantize);
            let accum = meta.query_advice(aux[0], Rotation::cur());
            let out_scale = meta.query_advice(scale, Rotation::cur());
            let output = meta.query_advice(quantized, Rotation::cur());
            let err = meta.query_advice(error, Rotation::cur());

            vec![s * (accum - output * out_scale - err)]
        });

        Int8QuantConfig {
            value,
            quantized,
            scale,
            zero_point,
            error,
            aux,
            int8_table,
            s_quantize,
            s_dequantize,
            s_int8_range,
            s_mul,
            s_add,
            s_requantize,
            _marker: PhantomData,
        }
    }

    /// Loads the INT8 range lookup table.
    pub fn load_int8_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "INT8 range table",
            |mut table| {
                for i in 0..INT8_RANGE {
                    table.assign_cell(
                        || format!("int8_{}", i),
                        self.config.int8_table,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Verifies a symmetric quantization.
    pub fn verify_symmetric_quantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        witness: &Int8QuantWitness<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_quantize.enable(region, row)?;
        self.config.s_int8_range.enable(region, row)?;

        region.assign_advice(|| "original", self.config.value, row, || Value::known(witness.original))?;
        region.assign_advice(|| "quantized", self.config.quantized, row, || Value::known(witness.quantized))?;
        region.assign_advice(|| "scale", self.config.scale, row, || Value::known(witness.scale))?;
        region.assign_advice(|| "zero_point", self.config.zero_point, row, || Value::known(F::ZERO))?;
        region.assign_advice(|| "error", self.config.error, row, || Value::known(witness.error))?;

        Ok(())
    }

    /// Verifies an asymmetric quantization.
    pub fn verify_asymmetric_quantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        witness: &Int8QuantWitness<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_quantize.enable(region, row)?;
        self.config.s_int8_range.enable(region, row)?;

        region.assign_advice(|| "original", self.config.value, row, || Value::known(witness.original))?;
        region.assign_advice(|| "quantized", self.config.quantized, row, || Value::known(witness.quantized))?;
        region.assign_advice(|| "scale", self.config.scale, row, || Value::known(witness.scale))?;
        region.assign_advice(|| "zero_point", self.config.zero_point, row, || Value::known(witness.zero_point))?;
        region.assign_advice(|| "error", self.config.error, row, || Value::known(witness.error))?;

        Ok(())
    }

    /// Verifies a quantized multiplication.
    pub fn verify_multiplication(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        a: Value<F>,
        b: Value<F>,
        result: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_mul.enable(region, row)?;

        region.assign_advice(|| "a", self.config.aux[0], row, || a)?;
        region.assign_advice(|| "b", self.config.aux[1], row, || b)?;
        region.assign_advice(|| "result", self.config.aux[2], row, || result)?;

        Ok(())
    }

    /// Verifies requantization from INT32 accumulator to INT8.
    pub fn verify_requantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        accumulator: Value<F>,
        output_scale: Value<F>,
        output: Value<F>,
        error: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_requantize.enable(region, row)?;
        self.config.s_int8_range.enable(region, row)?;

        region.assign_advice(|| "accumulator", self.config.aux[0], row, || accumulator)?;
        region.assign_advice(|| "output_scale", self.config.scale, row, || output_scale)?;
        region.assign_advice(|| "output", self.config.quantized, row, || output)?;
        region.assign_advice(|| "error", self.config.error, row, || error)?;

        Ok(())
    }
}

/// Circuit for INT8 quantized matrix multiplication.
#[derive(Clone)]
pub struct Int8MatMulCircuit<F: PrimeField> {
    /// Input matrix A (quantized INT8).
    pub a: Vec<Vec<i8>>,
    /// Input matrix B (quantized INT8).
    pub b: Vec<Vec<i8>>,
    /// Output matrix C (quantized INT8 after requantization).
    pub c: Vec<Vec<i8>>,
    /// INT32 accumulators (before requantization).
    pub accumulators: Vec<Vec<i32>>,
    /// Scale for A.
    pub scale_a: f64,
    /// Scale for B.
    pub scale_b: f64,
    /// Output scale.
    pub scale_c: f64,
    /// Requantization scale (scale_a * scale_b / scale_c).
    pub requant_scale: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int8MatMulCircuit<F> {
    fn default() -> Self {
        Self {
            a: vec![vec![0]],
            b: vec![vec![0]],
            c: vec![vec![0]],
            accumulators: vec![vec![0]],
            scale_a: 1.0,
            scale_b: 1.0,
            scale_c: 1.0,
            requant_scale: 1.0,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Int8MatMulCircuit<F> {
    /// Creates a new INT8 matmul circuit.
    pub fn new(
        a: Vec<Vec<i8>>,
        b: Vec<Vec<i8>>,
        scale_a: f64,
        scale_b: f64,
        scale_c: f64,
    ) -> Self {
        let m = a.len();
        let k = if m > 0 { a[0].len() } else { 0 };
        let n = if !b.is_empty() && !b[0].is_empty() { b[0].len() } else { 0 };

        // Compute INT32 accumulators
        let mut accumulators = vec![vec![0i32; n]; m];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    accumulators[i][j] += (a[i][kk] as i32) * (b[kk][j] as i32);
                }
            }
        }

        // Compute requantization
        let requant_scale = (scale_a * scale_b) / scale_c;
        let c: Vec<Vec<i8>> = accumulators
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&acc| {
                        let scaled = (acc as f64 * requant_scale).round();
                        scaled.clamp(INT8_MIN as f64, INT8_MAX as f64) as i8
                    })
                    .collect()
            })
            .collect();

        Self {
            a,
            b,
            c,
            accumulators,
            scale_a,
            scale_b,
            scale_c,
            requant_scale,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for Int8MatMulCircuit<F> {
    type Config = Int8QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int8QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = Int8QuantChip::new(config.clone());
        chip.load_int8_table(&mut layouter)?;

        let m = self.a.len();
        let k = if m > 0 { self.a[0].len() } else { 0 };
        let n = if !self.b.is_empty() && !self.b[0].is_empty() { self.b[0].len() } else { 0 };

        layouter.assign_region(
            || "int8_matmul",
            |mut region| {
                let mut row = 0;

                for i in 0..m {
                    for j in 0..n {
                        // Verify dot product computation
                        let mut acc = 0i32;
                        for kk in 0..k {
                            let a_val = self.a[i][kk];
                            let b_val = self.b[kk][j];
                            let prod = (a_val as i32) * (b_val as i32);

                            // Verify multiplication
                            chip.verify_multiplication(
                                &mut region,
                                row,
                                Value::known(Self::i8_to_field(a_val)),
                                Value::known(Self::i8_to_field(b_val)),
                                Value::known(Self::i32_to_field(prod)),
                            )?;
                            row += 1;

                            acc += prod;
                        }

                        // Verify accumulator matches
                        if acc != self.accumulators[i][j] {
                            return Err(ErrorFront::Synthesis);
                        }

                        // Verify requantization
                        // Constraint: accumulator = output * output_scale + error
                        // Since output = round(accumulator * requant_scale),
                        // we need output_scale = 1 / requant_scale
                        let output = self.c[i][j];
                        // Use fixed-point representation for output_scale
                        let fp_scale = 1000u64;
                        let output_scale_fp = if self.requant_scale.abs() > 1e-10 {
                            (fp_scale as f64 / self.requant_scale).round() as u64
                        } else {
                            fp_scale
                        };
                        // error = accumulator - output * output_scale
                        let error = (acc as i64) * (fp_scale as i64) - (output as i64) * (output_scale_fp as i64);

                        chip.verify_requantization(
                            &mut region,
                            row,
                            Value::known(Self::i32_to_field(acc) * F::from(fp_scale)),
                            Value::known(F::from(output_scale_fp)),
                            Value::known(Self::i8_to_field(output)),
                            Value::known(Self::i64_to_field(error)),
                        )?;
                        row += 1;
                    }
                }

                Ok(())
            },
        )
    }
}

impl<F: PrimeField> Int8MatMulCircuit<F> {
    fn i8_to_field(x: i8) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }

    fn i32_to_field(x: i32) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }

    fn i64_to_field(x: i64) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

/// Circuit for INT8 quantized dot product.
#[derive(Clone)]
pub struct Int8DotProductCircuit<F: PrimeField> {
    /// First vector (quantized INT8).
    pub a: Vec<i8>,
    /// Second vector (quantized INT8).
    pub b: Vec<i8>,
    /// Result (INT32 accumulator).
    pub result: i32,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int8DotProductCircuit<F> {
    fn default() -> Self {
        Self {
            a: vec![0],
            b: vec![0],
            result: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Int8DotProductCircuit<F> {
    /// Creates a new dot product circuit.
    pub fn new(a: Vec<i8>, b: Vec<i8>) -> Self {
        assert_eq!(a.len(), b.len());
        let result: i32 = a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum();
        Self {
            a,
            b,
            result,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for Int8DotProductCircuit<F> {
    type Config = Int8QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int8QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = Int8QuantChip::new(config.clone());
        chip.load_int8_table(&mut layouter)?;

        layouter.assign_region(
            || "int8_dot_product",
            |mut region| {
                let mut row = 0;
                let mut acc = 0i32;

                for (&a_val, &b_val) in self.a.iter().zip(self.b.iter()) {
                    let prod = (a_val as i32) * (b_val as i32);

                    chip.verify_multiplication(
                        &mut region,
                        row,
                        Value::known(Int8MatMulCircuit::<F>::i8_to_field(a_val)),
                        Value::known(Int8MatMulCircuit::<F>::i8_to_field(b_val)),
                        Value::known(Int8MatMulCircuit::<F>::i32_to_field(prod)),
                    )?;
                    row += 1;

                    acc += prod;
                }

                if acc != self.result {
                    return Err(ErrorFront::Synthesis);
                }

                Ok(())
            },
        )
    }
}

/// Circuit for INT8 quantization verification.
///
/// Verifies that a set of values are correctly quantized to INT8 format
/// with the given scale and computes quantization errors.
#[derive(Clone)]
pub struct Int8QuantCircuit<F: PrimeField> {
    /// Original values (fixed-point scaled).
    pub values: Vec<i64>,
    /// Quantized INT8 values.
    pub quantized: Vec<i8>,
    /// Scale factor (fixed-point).
    pub scale: u64,
    /// Quantization errors.
    pub errors: Vec<i64>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int8QuantCircuit<F> {
    fn default() -> Self {
        Self {
            values: vec![0],
            quantized: vec![0],
            scale: 1,
            errors: vec![0],
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Int8QuantCircuit<F> {
    /// Creates a new INT8 quantization circuit.
    pub fn new(values: Vec<f64>, fp_scale: u64) -> Self {
        let params = Int8SymmetricParams::from_range(
            values.iter().cloned().fold(f64::INFINITY, f64::min),
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );

        let quantized: Vec<i8> = values.iter().map(|&v| params.quantize(v)).collect();
        let values_fp: Vec<i64> = values.iter().map(|&v| (v * fp_scale as f64).round() as i64).collect();

        let scale_fp = (params.scale * fp_scale as f64).round() as u64;

        let errors: Vec<i64> = values.iter()
            .zip(quantized.iter())
            .map(|(&v, &q)| {
                let error = v - params.dequantize(q);
                (error * fp_scale as f64).round() as i64
            })
            .collect();

        Self {
            values: values_fp,
            quantized,
            scale: scale_fp,
            errors,
            _marker: PhantomData,
        }
    }

    fn i64_to_field(x: i64) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }

    fn i8_to_field(x: i8) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

impl<F: PrimeField> Circuit<F> for Int8QuantCircuit<F> {
    type Config = Int8QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int8QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = Int8QuantChip::new(config.clone());
        chip.load_int8_table(&mut layouter)?;

        layouter.assign_region(
            || "int8_quantization",
            |mut region| {
                for (i, ((&v, &q), &e)) in self.values.iter()
                    .zip(self.quantized.iter())
                    .zip(self.errors.iter())
                    .enumerate()
                {
                    let witness = Int8QuantWitness {
                        original: Self::i64_to_field(v),
                        quantized: Self::i8_to_field(q),
                        scale: F::from(self.scale),
                        zero_point: F::ZERO,
                        error: Self::i64_to_field(e),
                    };

                    chip.verify_symmetric_quantization(
                        &mut region,
                        i,
                        &witness,
                    )?;
                }
                Ok(())
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_symmetric_params() {
        let params = Int8SymmetricParams::from_range(-1.0, 1.0);

        // Quantize and dequantize
        let original = 0.5;
        let q = params.quantize(original);
        let dequant = params.dequantize(q);

        assert!((original - dequant).abs() <= params.max_error());
    }

    #[test]
    fn test_asymmetric_params() {
        let params = Int8AsymmetricParams::from_range(0.0, 1.0);

        // Quantize and dequantize
        let original = 0.5;
        let q = params.quantize(original);
        let dequant = params.dequantize(q);

        assert!((original - dequant).abs() <= params.max_error());
    }

    #[test]
    fn test_per_channel_params() {
        let ranges = vec![(-1.0, 1.0), (-2.0, 2.0), (-0.5, 0.5)];
        let params = PerChannelInt8Params::symmetric_from_ranges(&ranges);

        assert_eq!(params.num_channels, 3);

        // Channel 0: scale = 1/127
        let q0 = params.quantize(0.5, 0);
        let dq0 = params.dequantize(q0, 0);
        assert!((0.5 - dq0).abs() < 0.01);

        // Channel 1: scale = 2/127
        let q1 = params.quantize(1.0, 1);
        let dq1 = params.dequantize(q1, 1);
        assert!((1.0 - dq1).abs() < 0.02);
    }

    #[test]
    fn test_int8_dot_product_circuit() {
        let a = vec![1i8, 2, 3, 4];
        let b = vec![5i8, 6, 7, 8];
        // Dot product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

        let circuit = Int8DotProductCircuit::<Fr>::new(a, b);
        assert_eq!(circuit.result, 70);

        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_int8_matmul_circuit() {
        // 2x2 matrix multiplication
        let a = vec![
            vec![1i8, 2],
            vec![3, 4],
        ];
        let b = vec![
            vec![5i8, 6],
            vec![7, 8],
        ];

        let circuit = Int8MatMulCircuit::<Fr>::new(a, b, 0.01, 0.01, 0.0001);

        // Verify accumulators
        // C[0][0] = 1*5 + 2*7 = 19
        // C[0][1] = 1*6 + 2*8 = 22
        // C[1][0] = 3*5 + 4*7 = 43
        // C[1][1] = 3*6 + 4*8 = 50
        assert_eq!(circuit.accumulators[0][0], 19);
        assert_eq!(circuit.accumulators[0][1], 22);
        assert_eq!(circuit.accumulators[1][0], 43);
        assert_eq!(circuit.accumulators[1][1], 50);

        let prover = MockProver::run(12, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
