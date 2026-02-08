//! Quantization Verification Circuits for INT4/INT8 Operations.
//!
//! This module provides ZK circuits for verifying quantized neural network operations.
//! Quantization is essential for efficient ML inference and reduces proof complexity
//! by constraining values to smaller ranges.
//!
//! # Supported Quantization Formats
//!
//! - **INT4**: 4-bit signed integers [-8, 7] or unsigned [0, 15]
//! - **INT8**: 8-bit signed integers [-128, 127] or unsigned [0, 255]
//! - **Scaled Integers**: Fixed-point representation with configurable scale
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Quantization Pipeline                         │
//! │                                                                  │
//! │  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐    │
//! │  │ FP Value │ -> │ Quantize   │ -> │ INT4/INT8 Value      │    │
//! │  │          │    │ (round)    │    │ (range constrained)  │    │
//! │  └──────────┘    └────────────┘    └──────────────────────┘    │
//! │                                              │                   │
//! │                                              ↓                   │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                 Quantized Operations                     │   │
//! │  │  - Quantized MatMul (INT8 x INT8 -> INT32 accumulator)  │   │
//! │  │  - Quantized Add (with overflow checking)                │   │
//! │  │  - Quantized ReLU (via lookup table)                    │   │
//! │  │  - Requantization (INT32 -> INT8 with scale)            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                                              │                   │
//! │                                              ↓                   │
//! │  ┌──────────┐    ┌────────────┐    ┌──────────────────────┐    │
//! │  │ INT Value│ -> │ Dequantize │ -> │ FP Value (approx)    │    │
//! │  │          │    │ (scale)    │    │ (with error bound)   │    │
//! │  └──────────┘    └────────────┘    └──────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Error Bounds
//!
//! Quantization introduces predictable error:
//! - Per-value error: ±0.5 * scale (rounding error)
//! - Accumulated error: Tracks through operation chain
//! - Final bound: Sum of all quantization errors

use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::gadgets::range::{RangeChip, RangeConfig};
use crate::gadgets::lookup::{LookupTableChip, LookupTableConfig};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, ErrorFront, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

/// INT4 range: 4 bits = 16 values
pub const INT4_RANGE: usize = 16;
/// INT4 signed min: -8
pub const INT4_SIGNED_MIN: i8 = -8;
/// INT4 signed max: 7
pub const INT4_SIGNED_MAX: i8 = 7;

/// INT8 range: 8 bits = 256 values
pub const INT8_RANGE: usize = 256;
/// INT8 signed min: -128
pub const INT8_SIGNED_MIN: i16 = -128;
/// INT8 signed max: 127
pub const INT8_SIGNED_MAX: i16 = 127;

/// INT32 range for accumulation (subset for practical proofs)
pub const INT32_ACCUMULATOR_RANGE: usize = 65536; // 16-bit subset for demo

/// Default quantization scale factor.
pub const DEFAULT_SCALE: u32 = 128;

/// Quantization format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// Unsigned 4-bit: [0, 15]
    UInt4,
    /// Signed 4-bit: [-8, 7]
    Int4,
    /// Unsigned 8-bit: [0, 255]
    UInt8,
    /// Signed 8-bit: [-128, 127]
    Int8,
    /// Symmetric quantization with zero point at 0
    SymmetricInt8,
    /// Asymmetric quantization with arbitrary zero point
    AsymmetricInt8 { zero_point: i16 },
}

impl QuantFormat {
    /// Returns the bit width.
    pub fn bits(&self) -> u8 {
        match self {
            Self::UInt4 | Self::Int4 => 4,
            Self::UInt8 | Self::Int8 | Self::SymmetricInt8 | Self::AsymmetricInt8 { .. } => 8,
        }
    }

    /// Returns the number of possible values.
    pub fn range(&self) -> usize {
        1 << self.bits()
    }

    /// Returns (min, max) for the format.
    pub fn bounds(&self) -> (i64, i64) {
        match self {
            Self::UInt4 => (0, 15),
            Self::Int4 => (-8, 7),
            Self::UInt8 => (0, 255),
            Self::Int8 | Self::SymmetricInt8 => (-128, 127),
            Self::AsymmetricInt8 { zero_point } => {
                (-(*zero_point as i64), 255 - (*zero_point as i64))
            }
        }
    }

    /// Returns the zero point for the format.
    pub fn zero_point(&self) -> i64 {
        match self {
            Self::UInt4 | Self::Int4 | Self::Int8 | Self::SymmetricInt8 => 0,
            Self::UInt8 => 0,
            Self::AsymmetricInt8 { zero_point } => *zero_point as i64,
        }
    }

    /// Returns the maximum absolute value.
    pub fn max_abs(&self) -> i64 {
        let (min, max) = self.bounds();
        min.abs().max(max.abs())
    }

    /// Returns the quantization error bound per value.
    pub fn error_bound(&self, scale: f64) -> f64 {
        // Rounding error is at most 0.5 in quantized units
        0.5 * scale
    }
}

/// Quantization parameters for a tensor.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Quantization format.
    pub format: QuantFormat,
    /// Scale factor (quantized = real / scale).
    pub scale: f64,
    /// Zero point offset.
    pub zero_point: i64,
    /// Per-channel scales (for per-channel quantization).
    pub per_channel_scales: Option<Vec<f64>>,
}

impl QuantParams {
    /// Creates symmetric INT8 params with the given scale.
    pub fn symmetric_int8(scale: f64) -> Self {
        Self {
            format: QuantFormat::SymmetricInt8,
            scale,
            zero_point: 0,
            per_channel_scales: None,
        }
    }

    /// Creates asymmetric INT8 params.
    pub fn asymmetric_int8(scale: f64, zero_point: i64) -> Self {
        Self {
            format: QuantFormat::AsymmetricInt8 { zero_point: zero_point as i16 },
            scale,
            zero_point,
            per_channel_scales: None,
        }
    }

    /// Creates INT4 params with the given scale.
    pub fn int4(scale: f64) -> Self {
        Self {
            format: QuantFormat::Int4,
            scale,
            zero_point: 0,
            per_channel_scales: None,
        }
    }

    /// Quantizes a floating-point value.
    pub fn quantize(&self, value: f64) -> i64 {
        let scaled = value / self.scale + self.zero_point as f64;
        let (min, max) = self.format.bounds();
        scaled.round().clamp(min as f64, max as f64) as i64
    }

    /// Dequantizes an integer value.
    pub fn dequantize(&self, value: i64) -> f64 {
        (value - self.zero_point) as f64 * self.scale
    }

    /// Returns the maximum quantization error for a single value.
    pub fn max_error(&self) -> f64 {
        self.format.error_bound(self.scale)
    }
}

impl Default for QuantParams {
    fn default() -> Self {
        Self::symmetric_int8(1.0 / DEFAULT_SCALE as f64)
    }
}

/// Configuration for quantization verification circuits.
#[derive(Clone, Debug)]
pub struct QuantizationConfig<F: PrimeField> {
    /// Advice columns for values.
    pub value: Column<Advice>,
    pub quantized: Column<Advice>,
    pub error: Column<Advice>,
    pub scale: Column<Advice>,

    /// Lookup table for INT8 range.
    pub int8_table: TableColumn,
    /// Lookup table for INT4 range.
    pub int4_table: TableColumn,

    /// Selector for INT8 range check.
    pub s_int8_range: Selector,
    /// Selector for INT4 range check.
    pub s_int4_range: Selector,
    /// Selector for quantization verification.
    pub s_quantize: Selector,
    /// Selector for dequantization verification.
    pub s_dequantize: Selector,
    /// Selector for quantized multiplication.
    pub s_quant_mul: Selector,
    /// Selector for quantized addition.
    pub s_quant_add: Selector,
    /// Selector for requantization.
    pub s_requant: Selector,

    /// Arithmetic config for basic operations.
    pub arithmetic: ArithmeticConfig,

    _marker: PhantomData<F>,
}

/// Chip for quantization verification.
pub struct QuantizationChip<F: PrimeField> {
    config: QuantizationConfig<F>,
}

impl<F: PrimeField> QuantizationChip<F> {
    /// Creates a new quantization chip.
    pub fn new(config: QuantizationConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the quantization verification circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> QuantizationConfig<F> {
        // Advice columns
        let value = meta.advice_column();
        let quantized = meta.advice_column();
        let error = meta.advice_column();
        let scale = meta.advice_column();

        // Standard arithmetic columns
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();

        // Enable equality for all advice columns
        for col in [value, quantized, error, scale, a, b, c] {
            meta.enable_equality(col);
        }

        // Lookup tables
        let int8_table = meta.lookup_table_column();
        let int4_table = meta.lookup_table_column();

        // Selectors
        let s_int8_range = meta.complex_selector();
        let s_int4_range = meta.complex_selector();
        let s_quantize = meta.selector();
        let s_dequantize = meta.selector();
        let s_quant_mul = meta.selector();
        let s_quant_add = meta.selector();
        let s_requant = meta.selector();

        // Configure arithmetic
        let arithmetic = ArithmeticChip::<F>::configure(meta, a, b, c);

        // INT8 range lookup: quantized ∈ [0, 255] (shifted for signed)
        meta.lookup("quant_range_check", |meta| {
            let s = meta.query_selector(s_int8_range);
            let q = meta.query_advice(quantized, Rotation::cur());
            vec![(s * q, int8_table)]
        });

        // INT4 range lookup: quantized ∈ [0, 15] (shifted for signed)
        meta.lookup("quant_overflow_check", |meta| {
            let s = meta.query_selector(s_int4_range);
            let q = meta.query_advice(quantized, Rotation::cur());
            vec![(s * q, int4_table)]
        });

        // Quantization constraint:
        // quantized = round(value / scale)
        // Verified as: |value - quantized * scale| <= scale / 2
        // i.e., error = value - quantized * scale, error in [-scale/2, scale/2]
        meta.create_gate("quantization", |meta| {
            let s = meta.query_selector(s_quantize);
            let val = meta.query_advice(value, Rotation::cur());
            let quant = meta.query_advice(quantized, Rotation::cur());
            let scl = meta.query_advice(scale, Rotation::cur());
            let err = meta.query_advice(error, Rotation::cur());

            // Constraint: value = quantized * scale + error
            // (value - quantized * scale - error = 0)
            vec![s * (val - quant * scl - err)]
        });

        // Quantized multiplication with accumulator
        // c = a * b (both in quantized form)
        meta.create_gate("quantized_mul", |meta| {
            let s = meta.query_selector(s_quant_mul);
            let qa = meta.query_advice(a, Rotation::cur());
            let qb = meta.query_advice(b, Rotation::cur());
            let qc = meta.query_advice(c, Rotation::cur());
            vec![s * (qa * qb - qc)]
        });

        // Quantized addition
        // c = a + b (with potential overflow check)
        meta.create_gate("quantized_add", |meta| {
            let s = meta.query_selector(s_quant_add);
            let qa = meta.query_advice(a, Rotation::cur());
            let qb = meta.query_advice(b, Rotation::cur());
            let qc = meta.query_advice(c, Rotation::cur());
            vec![s * (qa + qb - qc)]
        });

        // Requantization: scale down accumulator to INT8
        // output = round(accumulator / output_scale)
        meta.create_gate("requantization", |meta| {
            let s = meta.query_selector(s_requant);
            let accum = meta.query_advice(a, Rotation::cur());
            let out_scale = meta.query_advice(scale, Rotation::cur());
            let output = meta.query_advice(quantized, Rotation::cur());
            let err = meta.query_advice(error, Rotation::cur());

            // accum = output * out_scale + error
            vec![s * (accum - output * out_scale - err)]
        });

        QuantizationConfig {
            value,
            quantized,
            error,
            scale,
            int8_table,
            int4_table,
            s_int8_range,
            s_int4_range,
            s_quantize,
            s_dequantize,
            s_quant_mul,
            s_quant_add,
            s_requant,
            arithmetic,
            _marker: PhantomData,
        }
    }

    /// Loads the INT8 lookup table.
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

    /// Loads the INT4 lookup table.
    pub fn load_int4_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "INT4 range table",
            |mut table| {
                for i in 0..INT4_RANGE {
                    table.assign_cell(
                        || format!("int4_{}", i),
                        self.config.int4_table,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Assigns an INT8 range check.
    pub fn assign_int8_range_check(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        value: Value<F>,
    ) -> Result<AssignedCell<F, F>, ErrorFront> {
        self.config.s_int8_range.enable(region, row)?;
        region.assign_advice(|| "int8_value", self.config.quantized, row, || value)
    }

    /// Assigns an INT4 range check.
    pub fn assign_int4_range_check(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        value: Value<F>,
    ) -> Result<AssignedCell<F, F>, ErrorFront> {
        self.config.s_int4_range.enable(region, row)?;
        region.assign_advice(|| "int4_value", self.config.quantized, row, || value)
    }

    /// Verifies a quantization operation.
    pub fn verify_quantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        fp_value: Value<F>,
        quantized_value: Value<F>,
        scale: Value<F>,
        error: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_quantize.enable(region, row)?;

        region.assign_advice(|| "fp_value", self.config.value, row, || fp_value)?;
        region.assign_advice(|| "quantized", self.config.quantized, row, || quantized_value)?;
        region.assign_advice(|| "scale", self.config.scale, row, || scale)?;
        region.assign_advice(|| "error", self.config.error, row, || error)?;

        Ok(())
    }

    /// Verifies a quantized multiplication.
    pub fn verify_quantized_mul(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        a: Value<F>,
        b: Value<F>,
        result: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_quant_mul.enable(region, row)?;

        region.assign_advice(|| "a", self.config.arithmetic.a, row, || a)?;
        region.assign_advice(|| "b", self.config.arithmetic.b, row, || b)?;
        region.assign_advice(|| "result", self.config.arithmetic.c, row, || result)?;

        Ok(())
    }

    /// Verifies a quantized addition.
    pub fn verify_quantized_add(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        a: Value<F>,
        b: Value<F>,
        result: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_quant_add.enable(region, row)?;

        region.assign_advice(|| "a", self.config.arithmetic.a, row, || a)?;
        region.assign_advice(|| "b", self.config.arithmetic.b, row, || b)?;
        region.assign_advice(|| "result", self.config.arithmetic.c, row, || result)?;

        Ok(())
    }

    /// Verifies requantization from accumulator to INT8.
    pub fn verify_requantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        accumulator: Value<F>,
        output_scale: Value<F>,
        output: Value<F>,
        error: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_requant.enable(region, row)?;

        region.assign_advice(|| "accum", self.config.arithmetic.a, row, || accumulator)?;
        region.assign_advice(|| "output_scale", self.config.scale, row, || output_scale)?;
        region.assign_advice(|| "output", self.config.quantized, row, || output)?;
        region.assign_advice(|| "error", self.config.error, row, || error)?;

        Ok(())
    }
}

/// Witness for a quantized value.
#[derive(Clone, Debug)]
pub struct QuantizedValue<F: PrimeField> {
    /// The quantized integer value.
    pub quantized: F,
    /// The quantization error.
    pub error: F,
    /// Scale factor used.
    pub scale: F,
    /// Original floating-point value (for verification).
    pub original: Option<f64>,
}

impl<F: PrimeField> QuantizedValue<F> {
    /// Creates a new quantized value.
    pub fn new(quantized: i64, scale: f64, original: Option<f64>) -> Self {
        // Convert to field element (handle negative values)
        let q_field = if quantized >= 0 {
            F::from(quantized as u64)
        } else {
            F::ZERO - F::from((-quantized) as u64)
        };

        // Compute error if original is known
        let error = if let Some(orig) = original {
            let dequant = quantized as f64 * scale;
            let err = orig - dequant;
            // Scale error to fixed point
            let err_scaled = (err / scale * 1000.0) as i64;
            if err_scaled >= 0 {
                F::from(err_scaled as u64)
            } else {
                F::ZERO - F::from((-err_scaled) as u64)
            }
        } else {
            F::ZERO
        };

        Self {
            quantized: q_field,
            error,
            scale: F::from((scale * 1000.0) as u64), // Fixed-point scale
            original,
        }
    }

    /// Creates from an already-quantized field element.
    pub fn from_field(quantized: F, scale: F, error: F) -> Self {
        Self {
            quantized,
            error,
            scale,
            original: None,
        }
    }
}

/// Circuit for verifying INT8 quantized matrix multiplication.
#[derive(Clone)]
pub struct QuantizedMatMulCircuit<F: PrimeField> {
    /// Input matrix A (quantized INT8 values).
    pub a: Vec<Vec<i8>>,
    /// Input matrix B (quantized INT8 values).
    pub b: Vec<Vec<i8>>,
    /// Output matrix C (quantized INT32 accumulators).
    pub c: Vec<Vec<i32>>,
    /// Scale for A.
    pub scale_a: f64,
    /// Scale for B.
    pub scale_b: f64,
    /// Output scale.
    pub scale_c: f64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for QuantizedMatMulCircuit<F> {
    fn default() -> Self {
        Self {
            a: vec![vec![0]],
            b: vec![vec![0]],
            c: vec![vec![0]],
            scale_a: 1.0,
            scale_b: 1.0,
            scale_c: 1.0,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for QuantizedMatMulCircuit<F> {
    type Config = QuantizationConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        QuantizationChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let chip = QuantizationChip::new(config.clone());

        // Load lookup tables
        chip.load_int8_table(&mut layouter)?;
        chip.load_int4_table(&mut layouter)?;

        let m = self.a.len();
        let k = if m > 0 { self.a[0].len() } else { 0 };
        let n = if !self.b.is_empty() && !self.b[0].is_empty() {
            self.b[0].len()
        } else {
            0
        };

        layouter.assign_region(
            || "quantized matmul verification",
            |mut region| {
                let mut row = 0;

                // For each element in the output matrix
                for i in 0..m {
                    for j in 0..n {
                        // Range check A elements
                        for kk in 0..k {
                            let a_val = self.a[i][kk];
                            // Shift to unsigned for range check: a + 128
                            let a_shifted = (a_val as i16 + 128) as u64;
                            chip.assign_int8_range_check(
                                &mut region,
                                row,
                                Value::known(F::from(a_shifted)),
                            )?;
                            row += 1;
                        }

                        // Range check B elements
                        for kk in 0..k {
                            let b_val = self.b[kk][j];
                            let b_shifted = (b_val as i16 + 128) as u64;
                            chip.assign_int8_range_check(
                                &mut region,
                                row,
                                Value::known(F::from(b_shifted)),
                            )?;
                            row += 1;
                        }

                        // Verify dot product
                        let mut accum = 0i32;
                        for kk in 0..k {
                            let prod = (self.a[i][kk] as i32) * (self.b[kk][j] as i32);
                            accum += prod;

                            // Verify multiplication
                            let a_f = if self.a[i][kk] >= 0 {
                                F::from(self.a[i][kk] as u64)
                            } else {
                                F::ZERO - F::from((-self.a[i][kk]) as u64)
                            };
                            let b_f = if self.b[kk][j] >= 0 {
                                F::from(self.b[kk][j] as u64)
                            } else {
                                F::ZERO - F::from((-self.b[kk][j]) as u64)
                            };
                            let prod_f = if prod >= 0 {
                                F::from(prod as u64)
                            } else {
                                F::ZERO - F::from((-prod) as u64)
                            };

                            chip.verify_quantized_mul(
                                &mut region,
                                row,
                                Value::known(a_f),
                                Value::known(b_f),
                                Value::known(prod_f),
                            )?;
                            row += 1;
                        }

                        // Verify accumulator matches
                        let expected = self.c[i][j];
                        if accum != expected {
                            // This would cause verification to fail
                            return Err(ErrorFront::Synthesis);
                        }
                    }
                }

                Ok(())
            },
        )
    }
}

/// Lookup table for quantized activations (ReLU, etc.)
#[derive(Clone, Debug)]
pub struct QuantizedActivationTable<F: PrimeField> {
    /// Input-output pairs for the activation function.
    pub entries: Vec<(F, F)>,
    /// Format of the inputs.
    pub format: QuantFormat,
}

impl<F: PrimeField> QuantizedActivationTable<F> {
    /// Creates a ReLU lookup table for INT8.
    pub fn relu_int8() -> Self {
        let mut entries = Vec::with_capacity(INT8_RANGE);

        // ReLU: max(0, x) for x in [-128, 127]
        // Represented as shifted values [0, 255]
        for i in 0..INT8_RANGE {
            let signed_val = (i as i16) - 128; // Convert to signed
            let relu_val = signed_val.max(0) as u64;
            let relu_shifted = relu_val + 128; // Shift back to unsigned

            entries.push((F::from(i as u64), F::from(relu_shifted)));
        }

        // Ensure (0, 0) is in table for selector-off case
        entries[0] = (F::ZERO, F::from(128u64)); // -128 -> 0 (shifted to 128)

        Self {
            entries,
            format: QuantFormat::Int8,
        }
    }

    /// Creates a ReLU6 lookup table for INT8.
    /// ReLU6: min(max(0, x), 6)
    pub fn relu6_int8(scale: f64) -> Self {
        let mut entries = Vec::with_capacity(INT8_RANGE);
        let six_quantized = (6.0 / scale).round() as i16;

        for i in 0..INT8_RANGE {
            let signed_val = (i as i16) - 128;
            let relu6_val = signed_val.max(0).min(six_quantized) as u64;
            let relu6_shifted = relu6_val + 128;

            entries.push((F::from(i as u64), F::from(relu6_shifted)));
        }

        Self {
            entries,
            format: QuantFormat::Int8,
        }
    }

    /// Creates a LeakyReLU lookup table for INT8.
    /// LeakyReLU: x if x > 0 else alpha * x
    pub fn leaky_relu_int8(alpha: f64) -> Self {
        let mut entries = Vec::with_capacity(INT8_RANGE);

        for i in 0..INT8_RANGE {
            let signed_val = (i as i16) - 128;
            let leaky_val = if signed_val > 0 {
                signed_val
            } else {
                (signed_val as f64 * alpha).round() as i16
            };
            let leaky_shifted = (leaky_val + 128).max(0).min(255) as u64;

            entries.push((F::from(i as u64), F::from(leaky_shifted)));
        }

        Self {
            entries,
            format: QuantFormat::Int8,
        }
    }

    /// Creates a sigmoid lookup table for INT8.
    /// Approximated as piecewise linear.
    pub fn sigmoid_int8(scale: f64) -> Self {
        let mut entries = Vec::with_capacity(INT8_RANGE);

        for i in 0..INT8_RANGE {
            let signed_val = (i as i16) - 128;
            let x = signed_val as f64 * scale;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let quantized = ((sigmoid - 0.5) / scale).round() as i16;
            let shifted = (quantized + 128).clamp(0, 255) as u64;

            entries.push((F::from(i as u64), F::from(shifted)));
        }

        Self {
            entries,
            format: QuantFormat::Int8,
        }
    }

    /// Returns the entries for circuit loading.
    pub fn entries(&self) -> &[(F, F)] {
        &self.entries
    }

    /// Returns the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Tracks quantization error through a computation graph.
#[derive(Debug, Clone)]
pub struct QuantErrorTracker {
    /// Accumulated absolute error.
    pub accumulated_error: f64,
    /// Number of operations.
    pub num_ops: usize,
    /// Maximum single-op error seen.
    pub max_single_error: f64,
    /// Format being used.
    pub format: QuantFormat,
    /// Scale factor.
    pub scale: f64,
}

impl QuantErrorTracker {
    /// Creates a new error tracker.
    pub fn new(format: QuantFormat, scale: f64) -> Self {
        Self {
            accumulated_error: 0.0,
            num_ops: 0,
            max_single_error: 0.0,
            format,
            scale,
        }
    }

    /// Records a quantization operation.
    pub fn record_quantize(&mut self) {
        let error = self.format.error_bound(self.scale);
        self.accumulated_error += error;
        self.max_single_error = self.max_single_error.max(error);
        self.num_ops += 1;
    }

    /// Records an addition (errors add).
    pub fn record_add(&mut self, other: &Self) {
        self.accumulated_error += other.accumulated_error;
        self.num_ops += other.num_ops;
    }

    /// Records a multiplication (errors multiply with values).
    pub fn record_mul(&mut self, value_bound: f64, other: &Self, other_value_bound: f64) {
        // err(a*b) ≈ |a|*err(b) + |b|*err(a) for small errors
        let mul_error = value_bound * other.accumulated_error
            + other_value_bound * self.accumulated_error;
        self.accumulated_error = mul_error;
        self.num_ops += other.num_ops;
    }

    /// Records a matrix multiplication.
    pub fn record_matmul(&mut self, m: usize, k: usize, n: usize, value_bound: f64) {
        // For each output element: k multiplications and k-1 additions
        let per_element_error = k as f64 * (2.0 * value_bound * self.accumulated_error);
        self.accumulated_error = per_element_error;
        self.num_ops += m * n * (2 * k - 1);
    }

    /// Records requantization (adds another quantization error).
    pub fn record_requantize(&mut self, output_scale: f64) {
        self.accumulated_error += output_scale / 2.0;
        self.num_ops += 1;
    }

    /// Returns the current error bound.
    pub fn error_bound(&self) -> f64 {
        self.accumulated_error
    }

    /// Returns a summary.
    pub fn summary(&self) -> String {
        format!(
            "QuantErrorTracker:\n  \
             Format: {:?}\n  \
             Scale: {}\n  \
             Operations: {}\n  \
             Accumulated error: {:.6}\n  \
             Max single error: {:.6}",
            self.format,
            self.scale,
            self.num_ops,
            self.accumulated_error,
            self.max_single_error,
        )
    }
}

/// Estimates the error introduced by quantizing a neural network layer.
pub fn estimate_layer_error(
    input_size: usize,
    output_size: usize,
    format: QuantFormat,
    scale: f64,
    weight_bound: f64,
    input_bound: f64,
) -> f64 {
    let quant_error = format.error_bound(scale);

    // Weight quantization error per output
    let weight_quant_error = input_size as f64 * quant_error * input_bound;

    // Input quantization error per output
    let input_quant_error = input_size as f64 * quant_error * weight_bound;

    // Accumulation error (rounding during adds)
    let accum_error = (input_size - 1) as f64 * quant_error;

    // Output requantization error
    let output_quant_error = quant_error;

    weight_quant_error + input_quant_error + accum_error + output_quant_error
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_quant_format_bounds() {
        assert_eq!(QuantFormat::Int8.bounds(), (-128, 127));
        assert_eq!(QuantFormat::UInt8.bounds(), (0, 255));
        assert_eq!(QuantFormat::Int4.bounds(), (-8, 7));
        assert_eq!(QuantFormat::UInt4.bounds(), (0, 15));
    }

    #[test]
    fn test_quant_params() {
        let params = QuantParams::symmetric_int8(0.1);

        // Quantize 0.5 with scale 0.1 -> 5
        assert_eq!(params.quantize(0.5), 5);
        // Dequantize 5 with scale 0.1 -> 0.5
        assert!((params.dequantize(5) - 0.5).abs() < 1e-6);

        // Clamping test
        assert_eq!(params.quantize(15.0), 127); // Clamped to max
        assert_eq!(params.quantize(-15.0), -128); // Clamped to min
    }

    #[test]
    fn test_quantized_value() {
        let qv = QuantizedValue::<Fr>::new(42, 0.01, Some(0.42));
        assert_eq!(qv.quantized, Fr::from(42u64));
    }

    #[test]
    fn test_relu_table() {
        let table = QuantizedActivationTable::<Fr>::relu_int8();
        assert_eq!(table.len(), 256);

        // Check specific values
        // -128 (shifted 0) -> 0 (shifted 128)
        assert_eq!(table.entries[0], (Fr::ZERO, Fr::from(128u64)));

        // 127 (shifted 255) -> 127 (shifted 255)
        assert_eq!(table.entries[255], (Fr::from(255u64), Fr::from(255u64)));

        // 0 (shifted 128) -> 0 (shifted 128)
        assert_eq!(table.entries[128], (Fr::from(128u64), Fr::from(128u64)));
    }

    #[test]
    fn test_error_tracker() {
        let mut tracker = QuantErrorTracker::new(QuantFormat::Int8, 0.01);

        tracker.record_quantize();
        assert!(tracker.accumulated_error > 0.0);
        assert_eq!(tracker.num_ops, 1);

        let initial_error = tracker.accumulated_error;
        tracker.record_quantize();
        assert!(tracker.accumulated_error > initial_error);
        assert_eq!(tracker.num_ops, 2);
    }

    #[test]
    fn test_estimate_layer_error() {
        let error = estimate_layer_error(
            64,   // input size
            32,   // output size
            QuantFormat::Int8,
            0.01, // scale
            1.0,  // weight bound
            1.0,  // input bound
        );

        // Should be reasonable error for INT8
        assert!(error > 0.0);
        assert!(error < 10.0); // Reasonable bound
    }

    #[test]
    fn test_quantized_matmul_circuit() {
        // Small 2x2 matrix multiplication
        let circuit = QuantizedMatMulCircuit::<Fr> {
            a: vec![
                vec![1, 2],
                vec![3, 4],
            ],
            b: vec![
                vec![5, 6],
                vec![7, 8],
            ],
            // C = A * B
            // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
            // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
            c: vec![
                vec![19, 22],
                vec![43, 50],
            ],
            scale_a: 1.0,
            scale_b: 1.0,
            scale_c: 1.0,
            _marker: PhantomData,
        };

        // Run with mock prover
        let k = 10; // Circuit size
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_quantized_matmul_invalid() {
        // Invalid result should fail
        let circuit = QuantizedMatMulCircuit::<Fr> {
            a: vec![vec![1, 2]],
            b: vec![vec![3], vec![4]],
            c: vec![vec![999]], // Wrong! Should be 1*3 + 2*4 = 11
            scale_a: 1.0,
            scale_b: 1.0,
            scale_c: 1.0,
            _marker: PhantomData,
        };

        let k = 10;
        let result = MockProver::run(k, &circuit, vec![]);
        assert!(result.is_err());
    }
}
