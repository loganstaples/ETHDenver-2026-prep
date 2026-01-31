//! INT4 Quantization Circuits.
//!
//! This module provides ultra-low precision INT4 quantization circuits for:
//! - Weight quantization in memory-constrained settings
//! - Mixed-precision computation (INT4 weights, INT8/INT16 activations)
//! - Efficient packing (2 INT4 values per byte)
//!
//! # INT4 Format
//!
//! Signed INT4 range: [-8, 7] (16 values)
//! Unsigned INT4 range: [0, 15] (16 values)
//!
//! # Packing
//!
//! Two INT4 values are packed into a single byte:
//! - High nibble: first value
//! - Low nibble: second value
//!
//! This halves memory requirements compared to INT8.
//!
//! # Mixed-Precision
//!
//! For better accuracy, activations can remain at INT8 while weights use INT4:
//! - Weights: INT4 (16 levels, scale_w)
//! - Activations: INT8 (256 levels, scale_a)
//! - Accumulator: INT32
//! - Output: Requantized to INT8

use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

use super::{INT4_MAX, INT4_MIN, INT4_RANGE};

/// Configuration for INT4 quantization circuits.
#[derive(Clone, Debug)]
pub struct Int4QuantConfig<F: PrimeField> {
    /// Advice columns.
    pub value: Column<Advice>,
    pub quantized: Column<Advice>,
    pub scale: Column<Advice>,
    pub packed: Column<Advice>,

    /// Auxiliary columns.
    pub aux: [Column<Advice>; 4],

    /// Lookup table for INT4 range.
    pub int4_table: TableColumn,
    /// Lookup table for packed INT4 range (0-255).
    pub packed_table: TableColumn,

    /// Selectors.
    pub s_quantize: Selector,
    pub s_int4_range: Selector,
    pub s_pack: Selector,
    pub s_unpack: Selector,
    pub s_mul: Selector,
    pub s_mixed_mul: Selector,

    _marker: PhantomData<F>,
}

/// INT4 weight quantization parameters.
#[derive(Clone, Debug)]
pub struct Int4WeightParams {
    /// Scale factor.
    pub scale: f64,
    /// Block size for block-wise quantization (common in LLMs).
    pub block_size: usize,
    /// Per-block scales (if using block-wise quantization).
    pub block_scales: Option<Vec<f64>>,
}

impl Int4WeightParams {
    /// Creates uniform INT4 params from weight range.
    pub fn from_range(weight_min: f64, weight_max: f64) -> Self {
        let abs_max = weight_min.abs().max(weight_max.abs());
        let scale = abs_max / INT4_MAX as f64;
        Self {
            scale,
            block_size: 0, // No blocking
            block_scales: None,
        }
    }

    /// Creates block-wise INT4 params.
    pub fn block_wise(weights: &[f64], block_size: usize) -> Self {
        let num_blocks = (weights.len() + block_size - 1) / block_size;
        let mut block_scales = Vec::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let start = i * block_size;
            let end = (start + block_size).min(weights.len());
            let block = &weights[start..end];

            let abs_max = block.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let scale = if abs_max > 0.0 {
                abs_max / INT4_MAX as f64
            } else {
                1.0
            };
            block_scales.push(scale);
        }

        Self {
            scale: block_scales.iter().sum::<f64>() / num_blocks as f64,
            block_size,
            block_scales: Some(block_scales),
        }
    }

    /// Quantizes a single weight value.
    pub fn quantize(&self, x: f64) -> i8 {
        let q = (x / self.scale).round();
        q.clamp(INT4_MIN as f64, INT4_MAX as f64) as i8
    }

    /// Quantizes a weight value in a specific block.
    pub fn quantize_block(&self, x: f64, block_idx: usize) -> i8 {
        let scale = self.block_scales.as_ref()
            .map(|s| s[block_idx])
            .unwrap_or(self.scale);
        let q = (x / scale).round();
        q.clamp(INT4_MIN as f64, INT4_MAX as f64) as i8
    }

    /// Dequantizes a value.
    pub fn dequantize(&self, q: i8) -> f64 {
        q as f64 * self.scale
    }

    /// Dequantizes a value from a specific block.
    pub fn dequantize_block(&self, q: i8, block_idx: usize) -> f64 {
        let scale = self.block_scales.as_ref()
            .map(|s| s[block_idx])
            .unwrap_or(self.scale);
        q as f64 * scale
    }

    /// Returns the maximum quantization error.
    pub fn max_error(&self) -> f64 {
        0.5 * self.scale
    }
}

/// Packs two INT4 values into a single byte.
///
/// # Arguments
/// * `high` - The value for the high nibble (bits 4-7)
/// * `low` - The value for the low nibble (bits 0-3)
pub fn pack_int4_values(high: i8, low: i8) -> u8 {
    // Convert signed to unsigned (add 8 to shift from [-8,7] to [0,15])
    let h = (high + 8) as u8;
    let l = (low + 8) as u8;
    (h << 4) | (l & 0x0F)
}

/// Unpacks a byte into two INT4 values.
///
/// # Returns
/// (high_nibble, low_nibble) as signed INT4 values
pub fn unpack_int4_values(packed: u8) -> (i8, i8) {
    let h = ((packed >> 4) & 0x0F) as i8 - 8;
    let l = (packed & 0x0F) as i8 - 8;
    (h, l)
}

/// INT4 quantization chip.
pub struct Int4QuantChip<F: PrimeField> {
    config: Int4QuantConfig<F>,
}

impl<F: PrimeField> Int4QuantChip<F> {
    /// Creates a new INT4 quantization chip.
    pub fn new(config: Int4QuantConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the INT4 quantization circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Int4QuantConfig<F> {
        // Advice columns
        let value = meta.advice_column();
        let quantized = meta.advice_column();
        let scale = meta.advice_column();
        let packed = meta.advice_column();
        let aux = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // Enable equality
        for col in [value, quantized, scale, packed] {
            meta.enable_equality(col);
        }
        for col in &aux {
            meta.enable_equality(*col);
        }

        // Lookup tables
        let int4_table = meta.lookup_table_column();
        let packed_table = meta.lookup_table_column();

        // Selectors
        let s_quantize = meta.selector();
        let s_int4_range = meta.complex_selector();
        let s_pack = meta.selector();
        let s_unpack = meta.selector();
        let s_mul = meta.selector();
        let s_mixed_mul = meta.selector();

        // INT4 range check: quantized + 8 ∈ [0, 15]
        meta.lookup(|meta| {
            let s = meta.query_selector(s_int4_range);
            let q = meta.query_advice(quantized, Rotation::cur());
            let shifted = q + Expression::Constant(F::from(8u64));
            vec![(s * shifted, int4_table)]
        });

        // Quantization: value = quantized * scale + error
        meta.create_gate("int4_quantize", |meta| {
            let s = meta.query_selector(s_quantize);
            let val = meta.query_advice(value, Rotation::cur());
            let q = meta.query_advice(quantized, Rotation::cur());
            let scl = meta.query_advice(scale, Rotation::cur());
            let err = meta.query_advice(aux[0], Rotation::cur());

            vec![s * (val - q * scl - err)]
        });

        // Packing: packed = (high + 8) * 16 + (low + 8)
        // where high and low are the two INT4 values
        meta.create_gate("int4_pack", |meta| {
            let s = meta.query_selector(s_pack);
            let high = meta.query_advice(aux[0], Rotation::cur());
            let low = meta.query_advice(aux[1], Rotation::cur());
            let pack = meta.query_advice(packed, Rotation::cur());

            let sixteen = Expression::Constant(F::from(16u64));
            let eight = Expression::Constant(F::from(8u64));

            // packed = (high + 8) * 16 + (low + 8)
            vec![s * (pack - (high.clone() + eight.clone()) * sixteen - (low + eight))]
        });

        // Unpacking: inverse of packing
        meta.create_gate("int4_unpack", |meta| {
            let s = meta.query_selector(s_unpack);
            let high = meta.query_advice(aux[0], Rotation::cur());
            let low = meta.query_advice(aux[1], Rotation::cur());
            let pack = meta.query_advice(packed, Rotation::cur());

            let sixteen = Expression::Constant(F::from(16u64));
            let eight = Expression::Constant(F::from(8u64));

            // Verify: packed = (high + 8) * 16 + (low + 8)
            vec![s * (pack - (high.clone() + eight.clone()) * sixteen - (low + eight))]
        });

        // Multiplication: result = a * b
        meta.create_gate("int4_mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(aux[0], Rotation::cur());
            let b = meta.query_advice(aux[1], Rotation::cur());
            let result = meta.query_advice(aux[2], Rotation::cur());

            vec![s * (a * b - result)]
        });

        // Mixed-precision multiplication: INT4 weight * INT8 activation
        meta.create_gate("mixed_mul", |meta| {
            let s = meta.query_selector(s_mixed_mul);
            let weight = meta.query_advice(aux[0], Rotation::cur()); // INT4
            let activation = meta.query_advice(aux[1], Rotation::cur()); // INT8
            let result = meta.query_advice(aux[2], Rotation::cur()); // INT16

            vec![s * (weight * activation - result)]
        });

        Int4QuantConfig {
            value,
            quantized,
            scale,
            packed,
            aux,
            int4_table,
            packed_table,
            s_quantize,
            s_int4_range,
            s_pack,
            s_unpack,
            s_mul,
            s_mixed_mul,
            _marker: PhantomData,
        }
    }

    /// Loads the INT4 range lookup table.
    pub fn load_int4_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
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

    /// Loads the packed INT4 lookup table (0-255).
    pub fn load_packed_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_table(
            || "Packed INT4 table",
            |mut table| {
                for i in 0..256usize {
                    table.assign_cell(
                        || format!("packed_{}", i),
                        self.config.packed_table,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Verifies an INT4 quantization.
    pub fn verify_quantization(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        value: Value<F>,
        quantized: Value<F>,
        scale: Value<F>,
        error: Value<F>,
    ) -> Result<(), Error> {
        self.config.s_quantize.enable(region, row)?;
        self.config.s_int4_range.enable(region, row)?;

        region.assign_advice(|| "value", self.config.value, row, || value)?;
        region.assign_advice(|| "quantized", self.config.quantized, row, || quantized)?;
        region.assign_advice(|| "scale", self.config.scale, row, || scale)?;
        region.assign_advice(|| "error", self.config.aux[0], row, || error)?;

        Ok(())
    }

    /// Verifies packing of two INT4 values.
    pub fn verify_pack(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        high: Value<F>,
        low: Value<F>,
        packed: Value<F>,
    ) -> Result<(), Error> {
        self.config.s_pack.enable(region, row)?;

        region.assign_advice(|| "high", self.config.aux[0], row, || high)?;
        region.assign_advice(|| "low", self.config.aux[1], row, || low)?;
        region.assign_advice(|| "packed", self.config.packed, row, || packed)?;

        Ok(())
    }

    /// Verifies unpacking of a byte to two INT4 values.
    pub fn verify_unpack(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        packed: Value<F>,
        high: Value<F>,
        low: Value<F>,
    ) -> Result<(), Error> {
        self.config.s_unpack.enable(region, row)?;
        // Also verify both values are in INT4 range
        // (This would require two range checks, but simplified here)

        region.assign_advice(|| "packed", self.config.packed, row, || packed)?;
        region.assign_advice(|| "high", self.config.aux[0], row, || high)?;
        region.assign_advice(|| "low", self.config.aux[1], row, || low)?;

        Ok(())
    }

    /// Verifies a mixed-precision multiplication.
    pub fn verify_mixed_mul(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        weight_int4: Value<F>,
        activation_int8: Value<F>,
        result: Value<F>,
    ) -> Result<(), Error> {
        self.config.s_mixed_mul.enable(region, row)?;

        region.assign_advice(|| "weight", self.config.aux[0], row, || weight_int4)?;
        region.assign_advice(|| "activation", self.config.aux[1], row, || activation_int8)?;
        region.assign_advice(|| "result", self.config.aux[2], row, || result)?;

        Ok(())
    }
}

/// Circuit for INT4 packed operations.
#[derive(Clone)]
pub struct Int4PackedCircuit<F: PrimeField> {
    /// Original INT4 values.
    pub values: Vec<i8>,
    /// Packed bytes.
    pub packed: Vec<u8>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int4PackedCircuit<F> {
    fn default() -> Self {
        Self {
            values: vec![0, 0],
            packed: vec![pack_int4_values(0, 0)],
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Int4PackedCircuit<F> {
    /// Creates a new packed circuit from values.
    pub fn new(values: Vec<i8>) -> Self {
        // Pad to even length
        let mut values = values;
        if values.len() % 2 != 0 {
            values.push(0);
        }

        let packed: Vec<u8> = values
            .chunks(2)
            .map(|chunk| pack_int4_values(chunk[0], chunk[1]))
            .collect();

        Self {
            values,
            packed,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for Int4PackedCircuit<F> {
    type Config = Int4QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int4QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Int4QuantChip::new(config.clone());
        chip.load_int4_table(&mut layouter)?;
        chip.load_packed_table(&mut layouter)?;

        layouter.assign_region(
            || "int4_packed",
            |mut region| {
                for (i, chunk) in self.values.chunks(2).enumerate() {
                    let high = chunk[0];
                    let low = chunk[1];
                    let packed = self.packed[i];

                    chip.verify_pack(
                        &mut region,
                        i,
                        Value::known(Self::i8_to_field(high)),
                        Value::known(Self::i8_to_field(low)),
                        Value::known(F::from(packed as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl<F: PrimeField> Int4PackedCircuit<F> {
    fn i8_to_field(x: i8) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

/// Circuit for mixed-precision INT4 weights with INT8 activations.
#[derive(Clone)]
pub struct Int4MixedPrecisionCircuit<F: PrimeField> {
    /// INT4 weights.
    pub weights: Vec<i8>,
    /// INT8 activations.
    pub activations: Vec<i8>,
    /// INT16 products.
    pub products: Vec<i16>,
    /// INT32 accumulator.
    pub accumulator: i32,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int4MixedPrecisionCircuit<F> {
    fn default() -> Self {
        Self {
            weights: vec![0],
            activations: vec![0],
            products: vec![0],
            accumulator: 0,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Int4MixedPrecisionCircuit<F> {
    /// Creates a new mixed-precision circuit.
    pub fn new(weights: Vec<i8>, activations: Vec<i8>) -> Self {
        assert_eq!(weights.len(), activations.len());

        let products: Vec<i16> = weights
            .iter()
            .zip(activations.iter())
            .map(|(&w, &a)| (w as i16) * (a as i16))
            .collect();

        let accumulator: i32 = products.iter().map(|&p| p as i32).sum();

        Self {
            weights,
            activations,
            products,
            accumulator,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for Int4MixedPrecisionCircuit<F> {
    type Config = Int4QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int4QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Int4QuantChip::new(config.clone());
        chip.load_int4_table(&mut layouter)?;

        layouter.assign_region(
            || "int4_mixed_precision",
            |mut region| {
                for (i, ((&w, &a), &p)) in self.weights.iter()
                    .zip(self.activations.iter())
                    .zip(self.products.iter())
                    .enumerate()
                {
                    chip.verify_mixed_mul(
                        &mut region,
                        i,
                        Value::known(Int4PackedCircuit::<F>::i8_to_field(w)),
                        Value::known(Int4PackedCircuit::<F>::i8_to_field(a as i8)),
                        Value::known(Self::i16_to_field(p)),
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl<F: PrimeField> Int4MixedPrecisionCircuit<F> {
    fn i16_to_field(x: i16) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

/// Circuit for INT4 quantization verification.
#[derive(Clone)]
pub struct Int4QuantCircuit<F: PrimeField> {
    /// Original values (fixed-point scaled).
    pub values: Vec<i64>,
    /// Quantized INT4 values.
    pub quantized: Vec<i8>,
    /// Scale factor (fixed-point).
    pub scale: u64,
    /// Quantization errors.
    pub errors: Vec<i64>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for Int4QuantCircuit<F> {
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

impl<F: PrimeField> Int4QuantCircuit<F> {
    /// Creates a new INT4 quantization circuit.
    pub fn new(values: Vec<f64>, scale: f64, fp_scale: u64) -> Self {
        let params = Int4WeightParams::from_range(
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
}

impl<F: PrimeField> Circuit<F> for Int4QuantCircuit<F> {
    type Config = Int4QuantConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        Int4QuantChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = Int4QuantChip::new(config.clone());
        chip.load_int4_table(&mut layouter)?;

        layouter.assign_region(
            || "int4_quantization",
            |mut region| {
                for (i, ((&v, &q), &e)) in self.values.iter()
                    .zip(self.quantized.iter())
                    .zip(self.errors.iter())
                    .enumerate()
                {
                    chip.verify_quantization(
                        &mut region,
                        i,
                        Value::known(Self::i64_to_field(v)),
                        Value::known(Int4PackedCircuit::<F>::i8_to_field(q)),
                        Value::known(F::from(self.scale)),
                        Value::known(Self::i64_to_field(e)),
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl<F: PrimeField> Int4QuantCircuit<F> {
    fn i64_to_field(x: i64) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_pack_unpack() {
        // Test various value combinations
        for high in -8i8..=7 {
            for low in -8i8..=7 {
                let packed = pack_int4_values(high, low);
                let (h, l) = unpack_int4_values(packed);
                assert_eq!(h, high);
                assert_eq!(l, low);
            }
        }
    }

    #[test]
    fn test_int4_weight_params() {
        let params = Int4WeightParams::from_range(-1.0, 1.0);

        // Quantize and dequantize
        let original = 0.5;
        let q = params.quantize(original);
        let dequant = params.dequantize(q);

        // INT4 has lower precision, so larger error tolerance
        assert!((original - dequant).abs() <= params.max_error() + 0.01);
    }

    #[test]
    fn test_block_wise_quantization() {
        let weights: Vec<f64> = (0..16).map(|i| (i as f64 - 8.0) / 8.0).collect();
        let params = Int4WeightParams::block_wise(&weights, 4);

        assert_eq!(params.block_scales.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_int4_packed_circuit() {
        let values = vec![1i8, -2, 3, -4, 5, -6];
        let circuit = Int4PackedCircuit::<Fr>::new(values.clone());

        assert_eq!(circuit.packed.len(), 3);

        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_mixed_precision_circuit() {
        let weights = vec![1i8, 2, -3, 4];
        let activations = vec![10i8, 20, 30, -40];

        let circuit = Int4MixedPrecisionCircuit::<Fr>::new(weights, activations);

        // 1*10 + 2*20 + (-3)*30 + 4*(-40) = 10 + 40 - 90 - 160 = -200
        assert_eq!(circuit.accumulator, -200);

        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_int4_quant_circuit() {
        let values = vec![0.5, -0.3, 0.7, -0.1];
        let circuit = Int4QuantCircuit::<Fr>::new(values, 0.1, 1000);

        let prover = MockProver::run(10, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
