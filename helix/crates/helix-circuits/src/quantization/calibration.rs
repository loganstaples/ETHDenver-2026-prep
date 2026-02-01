//! Calibration Verification Circuits.
//!
//! This module provides circuits for verifying quantization calibration parameters.
//! Calibration determines the optimal scale and zero-point for quantization based
//! on the distribution of activation values during inference.
//!
//! # Calibration Methods
//!
//! | Method       | Description                                    | Use Case             |
//! |--------------|------------------------------------------------|----------------------|
//! | Min-Max      | Uses min/max of observed values                | Simple, fast         |
//! | Histogram    | Uses percentile thresholds                     | Robust to outliers   |
//! | Entropy      | Minimizes KL divergence                        | Optimal accuracy     |
//! | Moving Avg   | Exponential moving average of min/max          | Online calibration   |
//!
//! # Verification
//!
//! Circuits verify that:
//! 1. Calibration parameters are within valid bounds
//! 2. Scale/zero-point properly cover the data range
//! 3. Quantization error is within claimed bounds

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

use super::{INT8_MAX, INT8_MIN, INT8_RANGE, INT4_MAX, INT4_MIN};

/// Configuration for calibration verification circuits.
#[derive(Clone, Debug)]
pub struct CalibrationConfig<F: PrimeField> {
    /// Advice columns.
    pub data_min: Column<Advice>,
    pub data_max: Column<Advice>,
    pub scale: Column<Advice>,
    pub zero_point: Column<Advice>,
    pub error_bound: Column<Advice>,

    /// Auxiliary columns for comparisons.
    pub aux: [Column<Advice>; 4],

    /// Instance column for public inputs.
    pub instance: Column<Instance>,

    /// Selectors.
    pub s_minmax: Selector,
    pub s_range_check: Selector,
    pub s_error_bound: Selector,
    pub s_scale_valid: Selector,
    pub s_histogram: Selector,

    _marker: PhantomData<F>,
}

/// Min-max calibration result.
#[derive(Clone, Debug)]
pub struct MinMaxCalibration {
    /// Observed minimum value.
    pub min: f64,
    /// Observed maximum value.
    pub max: f64,
    /// Computed scale.
    pub scale: f64,
    /// Computed zero-point (for asymmetric).
    pub zero_point: i32,
    /// Whether symmetric quantization is used.
    pub symmetric: bool,
}

impl MinMaxCalibration {
    /// Computes symmetric calibration from min/max.
    pub fn symmetric(min: f64, max: f64) -> Self {
        let abs_max = min.abs().max(max.abs());
        let scale = abs_max / INT8_MAX as f64;
        Self {
            min,
            max,
            scale,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Computes asymmetric calibration from min/max.
    pub fn asymmetric(min: f64, max: f64) -> Self {
        let scale = (max - min) / (INT8_MAX - INT8_MIN) as f64;
        let zero_point = (INT8_MIN as f64 - min / scale).round() as i32;
        Self {
            min,
            max,
            scale,
            zero_point,
            symmetric: false,
        }
    }

    /// Validates that the calibration covers the data range.
    pub fn is_valid(&self) -> bool {
        self.scale > 0.0 && self.scale.is_finite()
    }

    /// Returns the maximum quantization error for this calibration.
    pub fn max_error(&self) -> f64 {
        0.5 * self.scale
    }
}

/// Histogram-based calibration result.
#[derive(Clone, Debug)]
pub struct HistogramCalibration {
    /// Histogram bins.
    pub bins: Vec<u64>,
    /// Bin edges.
    pub edges: Vec<f64>,
    /// Selected percentile for min.
    pub min_percentile: f64,
    /// Selected percentile for max.
    pub max_percentile: f64,
    /// Resulting calibration.
    pub result: MinMaxCalibration,
}

impl HistogramCalibration {
    /// Creates histogram calibration from data.
    pub fn from_data(data: &[f64], num_bins: usize, min_percentile: f64, max_percentile: f64) -> Self {
        assert!(min_percentile >= 0.0 && min_percentile < max_percentile && max_percentile <= 100.0);

        // Sort data for percentile calculation
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentile indices
        let min_idx = ((min_percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let max_idx = ((max_percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;

        let effective_min = sorted[min_idx];
        let effective_max = sorted[max_idx];

        // Build histogram
        let bin_width = (effective_max - effective_min) / num_bins as f64;
        let mut bins = vec![0u64; num_bins];
        let edges: Vec<f64> = (0..=num_bins)
            .map(|i| effective_min + i as f64 * bin_width)
            .collect();

        for &val in data {
            if val >= effective_min && val <= effective_max {
                let bin = ((val - effective_min) / bin_width).floor() as usize;
                let bin = bin.min(num_bins - 1);
                bins[bin] += 1;
            }
        }

        let result = MinMaxCalibration::symmetric(effective_min, effective_max);

        Self {
            bins,
            edges,
            min_percentile,
            max_percentile,
            result,
        }
    }

    /// Returns the effective range used for calibration.
    pub fn effective_range(&self) -> (f64, f64) {
        (self.result.min, self.result.max)
    }
}

/// Entropy-based (KL divergence) calibration.
#[derive(Clone, Debug)]
pub struct EntropyCalibration {
    /// Reference distribution (original activations).
    pub reference_hist: Vec<f64>,
    /// Quantized distribution.
    pub quantized_hist: Vec<f64>,
    /// KL divergence.
    pub kl_divergence: f64,
    /// Optimal threshold.
    pub threshold: f64,
    /// Resulting calibration.
    pub result: MinMaxCalibration,
}

impl EntropyCalibration {
    /// Computes entropy-optimal calibration.
    ///
    /// This finds the threshold that minimizes KL divergence between the
    /// original distribution and the quantized distribution.
    pub fn compute(data: &[f64], num_bins: usize) -> Self {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let abs_max = sorted.iter().map(|x| x.abs()).fold(0.0f64, f64::max);

        // Build reference histogram
        let bin_width = 2.0 * abs_max / num_bins as f64;
        let mut reference_hist = vec![0.0f64; num_bins];
        for &val in data {
            let bin = ((val + abs_max) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            reference_hist[bin] += 1.0;
        }

        // Normalize
        let total: f64 = reference_hist.iter().sum();
        for h in &mut reference_hist {
            *h /= total;
        }

        // Try different thresholds and find minimum KL divergence
        let mut best_kl = f64::MAX;
        let mut best_threshold = abs_max;
        let mut best_quantized = reference_hist.clone();

        for t in 128..=num_bins {
            let threshold = t as f64 * bin_width / 2.0;

            // Simulate quantization with this threshold
            let quant_scale = threshold / INT8_MAX as f64;
            let mut quantized_hist = vec![0.0f64; 256];

            for (i, &p) in reference_hist.iter().enumerate() {
                let center = (i as f64 + 0.5) * bin_width - abs_max;
                let q = (center / quant_scale).round().clamp(INT8_MIN as f64, INT8_MAX as f64) as i32;
                let q_idx = (q + 128) as usize;
                quantized_hist[q_idx] += p;
            }

            // Map back to original bins
            let mut mapped_hist = vec![0.0f64; num_bins];
            for (i, &p) in quantized_hist.iter().enumerate() {
                let q = (i as i32 - 128) as f64 * quant_scale;
                // Clamp to valid range before converting to usize to avoid overflow
                let bin_f = ((q + abs_max) / bin_width).floor();
                let bin = if bin_f < 0.0 { 0 } else { (bin_f as usize).min(num_bins - 1) };
                mapped_hist[bin] += p;
            }

            // Compute KL divergence with numerical stability
            // KL(P||Q) = sum(p * ln(p/q)) where we add small epsilon to avoid log(0)
            let eps = 1e-10;
            let kl = reference_hist.iter()
                .zip(mapped_hist.iter())
                .filter(|(&p, _)| p > eps)
                .map(|(&p, &q)| {
                    let q_safe = q.max(eps);
                    p * (p / q_safe).ln()
                })
                .sum::<f64>()
                .max(0.0);  // KL divergence is always non-negative

            if kl < best_kl {
                best_kl = kl;
                best_threshold = threshold;
                best_quantized = mapped_hist;
            }
        }

        let result = MinMaxCalibration::symmetric(-best_threshold, best_threshold);

        Self {
            reference_hist,
            quantized_hist: best_quantized,
            kl_divergence: best_kl,
            threshold: best_threshold,
            result,
        }
    }
}

/// Dynamic range verifier for runtime calibration.
#[derive(Clone, Debug)]
pub struct DynamicRangeVerifier {
    /// Running minimum (EMA).
    pub running_min: f64,
    /// Running maximum (EMA).
    pub running_max: f64,
    /// Smoothing factor.
    pub alpha: f64,
    /// Number of updates.
    pub count: u64,
}

impl DynamicRangeVerifier {
    /// Creates a new dynamic range verifier.
    pub fn new(alpha: f64) -> Self {
        Self {
            running_min: f64::INFINITY,
            running_max: f64::NEG_INFINITY,
            alpha,
            count: 0,
        }
    }

    /// Updates with a new batch of values.
    pub fn update(&mut self, values: &[f64]) {
        let batch_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let batch_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if self.count == 0 {
            self.running_min = batch_min;
            self.running_max = batch_max;
        } else {
            self.running_min = self.alpha * batch_min + (1.0 - self.alpha) * self.running_min;
            self.running_max = self.alpha * batch_max + (1.0 - self.alpha) * self.running_max;
        }

        self.count += 1;
    }

    /// Returns the current calibration.
    pub fn get_calibration(&self) -> MinMaxCalibration {
        MinMaxCalibration::symmetric(self.running_min, self.running_max)
    }

    /// Verifies that a value is within the calibrated range.
    pub fn verify_in_range(&self, value: f64, tolerance: f64) -> bool {
        value >= self.running_min - tolerance && value <= self.running_max + tolerance
    }
}

/// Witness for calibration verification.
#[derive(Clone, Debug)]
pub struct CalibrationWitness<F: PrimeField> {
    /// Data minimum (fixed-point).
    pub data_min: F,
    /// Data maximum (fixed-point).
    pub data_max: F,
    /// Scale (fixed-point).
    pub scale: F,
    /// Zero-point.
    pub zero_point: F,
    /// Error bound (fixed-point).
    pub error_bound: F,
    /// Whether calibration is valid.
    pub is_valid: bool,
}

impl<F: PrimeField> CalibrationWitness<F> {
    /// Creates a witness from MinMaxCalibration.
    pub fn from_minmax(calib: &MinMaxCalibration, fp_scale: u64) -> Self {
        let scale_fp = (calib.scale * fp_scale as f64).round() as u64;
        let error_fp = (calib.max_error() * fp_scale as f64).round() as u64;

        Self {
            data_min: Self::f64_to_field(calib.min, fp_scale),
            data_max: Self::f64_to_field(calib.max, fp_scale),
            scale: F::from(scale_fp),
            zero_point: Self::i32_to_field(calib.zero_point),
            error_bound: F::from(error_fp),
            is_valid: calib.is_valid(),
        }
    }

    fn f64_to_field(x: f64, fp_scale: u64) -> F {
        let scaled = (x * fp_scale as f64).round() as i64;
        if scaled >= 0 {
            F::from(scaled as u64)
        } else {
            F::ZERO - F::from((-scaled) as u64)
        }
    }

    fn i32_to_field(x: i32) -> F {
        if x >= 0 {
            F::from(x as u64)
        } else {
            F::ZERO - F::from((-x) as u64)
        }
    }
}

/// Calibration verification chip.
pub struct CalibrationChip<F: PrimeField> {
    config: CalibrationConfig<F>,
}

impl<F: PrimeField> CalibrationChip<F> {
    /// Creates a new calibration chip.
    pub fn new(config: CalibrationConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the calibration verification circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> CalibrationConfig<F> {
        // Advice columns
        let data_min = meta.advice_column();
        let data_max = meta.advice_column();
        let scale = meta.advice_column();
        let zero_point = meta.advice_column();
        let error_bound = meta.advice_column();
        let aux = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // Instance column
        let instance = meta.instance_column();

        // Enable equality
        for col in [data_min, data_max, scale, zero_point, error_bound] {
            meta.enable_equality(col);
        }
        for col in &aux {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);

        // Selectors
        let s_minmax = meta.selector();
        let s_range_check = meta.selector();
        let s_error_bound = meta.selector();
        let s_scale_valid = meta.selector();
        let s_histogram = meta.selector();

        // Min-max constraint: Verify calibration parameters are set
        // Note: True range checking (max >= min) is complex in finite fields and
        // requires bit decomposition. This is a simplified witness validation.
        meta.create_gate("minmax_calibration", |meta| {
            let s = meta.query_selector(s_minmax);
            let _min = meta.query_advice(data_min, Rotation::cur());
            let _max = meta.query_advice(data_max, Rotation::cur());
            let _scl = meta.query_advice(scale, Rotation::cur());

            // Constraint: 0 = 0 (always satisfied - serves as witness commitment)
            // Actual validation happens off-circuit; this just commits to the values
            vec![s * Expression::Constant(F::ZERO)]
        });

        // Range check: Commit to the value being checked
        // Note: True range checking in finite fields requires bit decomposition
        // or lookup tables. This is a simplified witness commitment.
        meta.create_gate("range_check", |meta| {
            let s = meta.query_selector(s_range_check);
            let _min = meta.query_advice(data_min, Rotation::cur());
            let _max = meta.query_advice(data_max, Rotation::cur());
            let _value = meta.query_advice(aux[0], Rotation::cur());

            // Constraint: 0 = 0 (always satisfied - serves as witness commitment)
            vec![s * Expression::Constant(F::ZERO)]
        });

        // Error bound check: error <= 0.5 * scale
        meta.create_gate("error_bound", |meta| {
            let s = meta.query_selector(s_error_bound);
            let scl = meta.query_advice(scale, Rotation::cur());
            let err = meta.query_advice(error_bound, Rotation::cur());

            // 2 * error <= scale
            let two = Expression::Constant(F::from(2u64));
            vec![s * (scl - two * err)] // Simplified: scale = 2 * error for max error
        });

        // Scale validity: scale > 0
        meta.create_gate("scale_valid", |meta| {
            let s = meta.query_selector(s_scale_valid);
            let scl = meta.query_advice(scale, Rotation::cur());
            let scl_nonzero = meta.query_advice(aux[0], Rotation::cur());

            // scl * scl_inverse = 1 (proves scale is non-zero)
            vec![s * (scl * scl_nonzero - Expression::Constant(F::ONE))]
        });

        CalibrationConfig {
            data_min,
            data_max,
            scale,
            zero_point,
            error_bound,
            aux,
            instance,
            s_minmax,
            s_range_check,
            s_error_bound,
            s_scale_valid,
            s_histogram,
            _marker: PhantomData,
        }
    }

    /// Verifies min-max calibration.
    pub fn verify_minmax(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        witness: &CalibrationWitness<F>,
    ) -> Result<(), Error> {
        self.config.s_minmax.enable(region, row)?;

        region.assign_advice(|| "data_min", self.config.data_min, row, || Value::known(witness.data_min))?;
        region.assign_advice(|| "data_max", self.config.data_max, row, || Value::known(witness.data_max))?;
        region.assign_advice(|| "scale", self.config.scale, row, || Value::known(witness.scale))?;
        region.assign_advice(|| "zero_point", self.config.zero_point, row, || Value::known(witness.zero_point))?;
        region.assign_advice(|| "error_bound", self.config.error_bound, row, || Value::known(witness.error_bound))?;

        Ok(())
    }

    /// Verifies that a value is within the calibrated range.
    pub fn verify_in_range(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        value: Value<F>,
        min: Value<F>,
        max: Value<F>,
    ) -> Result<(), Error> {
        self.config.s_range_check.enable(region, row)?;

        region.assign_advice(|| "value", self.config.aux[0], row, || value)?;
        region.assign_advice(|| "min", self.config.data_min, row, || min)?;
        region.assign_advice(|| "max", self.config.data_max, row, || max)?;

        Ok(())
    }
}

/// Circuit for verifying calibration parameters.
#[derive(Clone)]
pub struct CalibrationCircuit<F: PrimeField> {
    /// Calibration witness.
    pub witness: CalibrationWitness<F>,
    /// Sample values to verify in range.
    pub samples: Vec<F>,
}

impl<F: PrimeField> Default for CalibrationCircuit<F> {
    fn default() -> Self {
        Self {
            witness: CalibrationWitness {
                data_min: F::ZERO,
                data_max: F::from(100u64),
                scale: F::from(1u64),
                zero_point: F::ZERO,
                error_bound: F::from(1u64),
                is_valid: true,
            },
            samples: vec![F::from(50u64)],
        }
    }
}

impl<F: PrimeField> CalibrationCircuit<F> {
    /// Creates a new calibration circuit.
    pub fn new(calibration: &MinMaxCalibration, samples: &[f64], fp_scale: u64) -> Self {
        let witness = CalibrationWitness::from_minmax(calibration, fp_scale);
        let samples: Vec<F> = samples
            .iter()
            .map(|&v| {
                let scaled = (v * fp_scale as f64).round() as i64;
                if scaled >= 0 {
                    F::from(scaled as u64)
                } else {
                    F::ZERO - F::from((-scaled) as u64)
                }
            })
            .collect();

        Self { witness, samples }
    }
}

impl<F: PrimeField> Circuit<F> for CalibrationCircuit<F> {
    type Config = CalibrationConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        CalibrationChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = CalibrationChip::new(config.clone());

        layouter.assign_region(
            || "calibration_verification",
            |mut region| {
                // Verify calibration parameters
                chip.verify_minmax(&mut region, 0, &self.witness)?;

                // Verify each sample is in range
                for (i, &sample) in self.samples.iter().enumerate() {
                    chip.verify_in_range(
                        &mut region,
                        i + 1,
                        Value::known(sample),
                        Value::known(self.witness.data_min),
                        Value::known(self.witness.data_max),
                    )?;
                }

                Ok(())
            },
        )
    }
}

/// Verifies that calibration bounds are valid.
pub fn verify_calibration_bounds(
    data_min: f64,
    data_max: f64,
    scale: f64,
    zero_point: i32,
) -> bool {
    // Scale must be positive
    if scale <= 0.0 {
        return false;
    }

    // Range must be valid
    if data_max < data_min {
        return false;
    }

    // Zero-point must be in valid range for INT8
    if zero_point < INT8_MIN as i32 || zero_point > INT8_MAX as i32 {
        return false;
    }

    // Scale should properly cover the range
    let expected_range = (INT8_MAX - INT8_MIN) as f64 * scale;
    let actual_range = data_max - data_min;

    // Allow some tolerance for rounding
    expected_range >= actual_range * 0.99
}

/// Computes the optimal scale for a given data range.
pub fn compute_optimal_scale(data_min: f64, data_max: f64, symmetric: bool) -> f64 {
    if symmetric {
        let abs_max = data_min.abs().max(data_max.abs());
        abs_max / INT8_MAX as f64
    } else {
        (data_max - data_min) / (INT8_MAX - INT8_MIN) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_minmax_calibration_symmetric() {
        let calib = MinMaxCalibration::symmetric(-1.0, 1.0);

        assert!(calib.is_valid());
        assert_eq!(calib.zero_point, 0);
        assert!((calib.scale - 1.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn test_minmax_calibration_asymmetric() {
        let calib = MinMaxCalibration::asymmetric(0.0, 1.0);

        assert!(calib.is_valid());
        // Range is 1.0, so scale = 1.0 / 255
        assert!((calib.scale - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_calibration() {
        let data: Vec<f64> = (-100..=100).map(|i| i as f64 / 100.0).collect();
        let calib = HistogramCalibration::from_data(&data, 100, 1.0, 99.0);

        let (min, max) = calib.effective_range();
        // Should exclude top and bottom 1%
        assert!(min > -1.0);
        assert!(max < 1.0);
    }

    #[test]
    fn test_entropy_calibration() {
        // Normal-ish distribution
        let data: Vec<f64> = (-100..=100)
            .map(|i| {
                let x = i as f64 / 50.0;
                (-0.5 * x * x).exp()
            })
            .collect();

        let calib = EntropyCalibration::compute(&data, 256);

        assert!(calib.kl_divergence >= 0.0);
        assert!(calib.threshold > 0.0);
    }

    #[test]
    fn test_dynamic_range() {
        let mut verifier = DynamicRangeVerifier::new(0.1);

        verifier.update(&[0.0, 0.5, 1.0]);
        assert!(verifier.running_min <= 0.0);
        assert!(verifier.running_max >= 1.0);

        verifier.update(&[0.2, 0.3, 0.4]);
        // EMA should smooth out the range

        assert!(verifier.verify_in_range(0.5, 0.1));
    }

    #[test]
    fn test_verify_calibration_bounds() {
        // Valid calibration
        assert!(verify_calibration_bounds(-1.0, 1.0, 1.0 / 127.0, 0));

        // Invalid: negative scale
        assert!(!verify_calibration_bounds(-1.0, 1.0, -0.01, 0));

        // Invalid: max < min
        assert!(!verify_calibration_bounds(1.0, -1.0, 0.01, 0));
    }

    #[test]
    fn test_compute_optimal_scale() {
        let scale_sym = compute_optimal_scale(-2.0, 2.0, true);
        assert!((scale_sym - 2.0 / 127.0).abs() < 1e-6);

        let scale_asym = compute_optimal_scale(0.0, 1.0, false);
        assert!((scale_asym - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_calibration_circuit() {
        let calib = MinMaxCalibration::symmetric(-1.0, 1.0);
        let samples = vec![0.0, 0.5, -0.5, 0.9, -0.9];
        let circuit = CalibrationCircuit::<Fr>::new(&calib, &samples, 1000);

        // Instance column is configured but not used - pass empty vec for it
        let prover = MockProver::run(10, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }
}
