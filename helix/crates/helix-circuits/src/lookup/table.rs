//! Plookup-Style Lookup Table Infrastructure.
//!
//! This module provides a production-grade implementation of lookup tables
//! using the plookup protocol, optimized for neural network activation functions.
//!
//! # Key Features
//!
//! - **Multi-column lookups**: Support for tables with multiple input/output columns
//! - **Batch optimization**: Amortized cost when performing many lookups
//! - **Preprocessing**: Constant tables are preprocessed for optimal performance
//! - **Error tracking**: Lookup error bounds are tracked for approximate proofs
//!
//! # Theory
//!
//! Plookup reduces lookup verification to polynomial identity checks:
//! - Prover commits to sorted version of lookups + table
//! - Verifier checks grand product argument
//! - Cost: O(n log n) for n lookups vs O(n²) for naive approaches

use halo2_proofs::{
    circuit::{Layouter, Region, Value},
    plonk::{
        Advice, Column, ConstraintSystem, ErrorFront, Fixed, Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Error types for lookup operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// Value not found in lookup table.
    ValueNotInTable { input: String },
    /// Table size exceeds maximum.
    TableTooLarge { size: usize, max: usize },
    /// Invalid table configuration.
    InvalidConfiguration { reason: String },
    /// Circuit synthesis error.
    SynthesisError { message: String },
}

impl std::fmt::Display for LookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ValueNotInTable { input } => write!(f, "Value {} not found in lookup table", input),
            Self::TableTooLarge { size, max } => write!(f, "Table size {} exceeds maximum {}", size, max),
            Self::InvalidConfiguration { reason } => write!(f, "Invalid configuration: {}", reason),
            Self::SynthesisError { message } => write!(f, "Synthesis error: {}", message),
        }
    }
}

impl std::error::Error for LookupError {}

/// Statistics for lookup table usage.
#[derive(Debug, Clone, Default)]
pub struct LookupStats {
    /// Total number of lookups performed.
    pub total_lookups: usize,
    /// Number of unique values looked up.
    pub unique_lookups: usize,
    /// Cache hit rate (0.0 to 1.0).
    pub cache_hit_rate: f64,
    /// Average lookup time in microseconds.
    pub avg_lookup_time_us: f64,
    /// Table size in entries.
    pub table_size: usize,
    /// Memory usage in bytes.
    pub memory_bytes: usize,
}

/// Configuration for a plookup table.
#[derive(Clone, Debug)]
pub struct PlookupTableConfig<F: PrimeField> {
    /// Table columns for inputs (can be multiple for multi-input lookups).
    pub table_inputs: Vec<TableColumn>,
    /// Table columns for outputs.
    pub table_outputs: Vec<TableColumn>,
    /// Advice columns for lookup inputs.
    pub advice_inputs: Vec<Column<Advice>>,
    /// Advice columns for lookup outputs.
    pub advice_outputs: Vec<Column<Advice>>,
    /// Selector to enable the lookup.
    pub s_lookup: Selector,
    /// Optional fixed column for table index (for debugging).
    pub table_index: Option<Column<Fixed>>,
    _marker: PhantomData<F>,
}

/// Core plookup table structure.
///
/// This provides the foundation for all lookup-based activation functions.
#[derive(Clone, Debug)]
pub struct PlookupTable<F: PrimeField> {
    /// Precomputed table entries as (inputs, outputs) tuples.
    entries: Vec<(Vec<F>, Vec<F>)>,
    /// Lookup cache for fast repeated lookups.
    cache: HashMap<Vec<u64>, Vec<F>>,
    /// Table statistics.
    stats: LookupStats,
    /// Error bound for this table (maximum approximation error).
    error_bound: F,
    /// Scale factor for fixed-point representation.
    scale: u64,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> PlookupTable<F> {
    /// Creates a new empty plookup table.
    pub fn new(scale: u64) -> Self {
        Self {
            entries: Vec::new(),
            cache: HashMap::new(),
            stats: LookupStats::default(),
            error_bound: F::ZERO,
            scale,
            _marker: PhantomData,
        }
    }

    /// Creates a plookup table from a function.
    ///
    /// # Arguments
    /// * `range` - The input range (e.g., 0..256 for INT8)
    /// * `f` - Function mapping input to output
    /// * `scale` - Fixed-point scale factor
    pub fn from_function<G>(range: std::ops::Range<i64>, f: G, scale: u64) -> Self
    where
        G: Fn(i64) -> f64,
    {
        let mut table = Self::new(scale);
        let scale_f = scale as f64;
        let mut max_error = 0.0f64;

        for x in range {
            let exact = f(x);
            let quantized = (exact * scale_f).round() as i64;
            let error = (exact - (quantized as f64 / scale_f)).abs();
            max_error = max_error.max(error);

            let input = if x >= 0 {
                F::from(x as u64)
            } else {
                F::ZERO - F::from((-x) as u64)
            };

            let output = if quantized >= 0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            table.add_entry(vec![input], vec![output]);
        }

        // Set error bound based on maximum observed error
        table.error_bound = F::from((max_error * scale_f).ceil() as u64);
        table.stats.table_size = table.entries.len();

        table
    }

    /// Creates a table for a piecewise-linear function.
    pub fn from_piecewise_linear(
        breakpoints: &[(i64, f64)],
        range: std::ops::Range<i64>,
        scale: u64,
    ) -> Self {
        let mut table = Self::new(scale);
        let scale_f = scale as f64;

        for x in range {
            // Find the appropriate segment
            let y = Self::interpolate_piecewise(breakpoints, x as f64);
            let quantized = (y * scale_f).round() as i64;

            let input = if x >= 0 {
                F::from(x as u64)
            } else {
                F::ZERO - F::from((-x) as u64)
            };

            let output = if quantized >= 0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            table.add_entry(vec![input], vec![output]);
        }

        table.stats.table_size = table.entries.len();
        table
    }

    /// Interpolates a piecewise linear function.
    fn interpolate_piecewise(breakpoints: &[(i64, f64)], x: f64) -> f64 {
        if breakpoints.is_empty() {
            return 0.0;
        }

        // Find the segment containing x
        for i in 0..breakpoints.len() - 1 {
            let (x0, y0) = breakpoints[i];
            let (x1, y1) = breakpoints[i + 1];

            if x >= x0 as f64 && x <= x1 as f64 {
                // Linear interpolation
                let t = (x - x0 as f64) / (x1 - x0) as f64;
                return y0 + t * (y1 - y0);
            }
        }

        // Extrapolate from last segment
        let (x0, y0) = breakpoints[breakpoints.len() - 2];
        let (x1, y1) = breakpoints[breakpoints.len() - 1];
        let slope = (y1 - y0) / (x1 - x0) as f64;
        y1 + slope * (x - x1 as f64)
    }

    /// Adds a single entry to the table.
    pub fn add_entry(&mut self, inputs: Vec<F>, outputs: Vec<F>) {
        // Add to cache using repr conversion
        let cache_key: Vec<u64> = inputs.iter()
            .map(|f| {
                let repr = f.to_repr();
                let bytes = repr.as_ref();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8]))
            })
            .collect();
        self.cache.insert(cache_key, outputs.clone());

        self.entries.push((inputs, outputs));
    }

    /// Looks up a value in the table.
    pub fn lookup(&self, inputs: &[F]) -> Option<Vec<F>> {
        let cache_key: Vec<u64> = inputs.iter()
            .map(|f| {
                let repr = f.to_repr();
                let bytes = repr.as_ref();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0; 8]))
            })
            .collect();
        self.cache.get(&cache_key).cloned()
    }

    /// Returns the number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the entries as a slice.
    pub fn entries(&self) -> &[(Vec<F>, Vec<F>)] {
        &self.entries
    }

    /// Returns two-column entries for simple input→output tables.
    pub fn two_column_entries(&self) -> Vec<(F, F)> {
        self.entries
            .iter()
            .filter_map(|(inputs, outputs)| {
                if inputs.len() == 1 && outputs.len() == 1 {
                    Some((inputs[0], outputs[0]))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the error bound for this table.
    pub fn error_bound(&self) -> F {
        self.error_bound
    }

    /// Sets the error bound for this table.
    pub fn set_error_bound(&mut self, bound: F) {
        self.error_bound = bound;
    }

    /// Returns the scale factor.
    pub fn scale(&self) -> u64 {
        self.scale
    }

    /// Returns statistics about the table.
    pub fn stats(&self) -> &LookupStats {
        &self.stats
    }
}

/// Plookup chip for performing lookups in circuits.
pub struct PlookupChip<F: PrimeField> {
    config: PlookupTableConfig<F>,
}

impl<F: PrimeField> PlookupChip<F> {
    /// Creates a new plookup chip.
    pub fn new(config: PlookupTableConfig<F>) -> Self {
        Self { config }
    }

    /// Configures a simple two-column lookup.
    pub fn configure_simple(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> PlookupTableConfig<F> {
        let table_input = meta.lookup_table_column();
        let table_output = meta.lookup_table_column();
        let s_lookup = meta.complex_selector();

        meta.lookup("table_lookup", |meta| {
            let s = meta.query_selector(s_lookup);
            let input = meta.query_advice(advice_input, Rotation::cur());
            let output = meta.query_advice(advice_output, Rotation::cur());

            // When selector is off, expressions evaluate to 0
            // Table must contain (0, 0) as first entry
            vec![
                (s.clone() * input, table_input),
                (s * output, table_output),
            ]
        });

        PlookupTableConfig {
            table_inputs: vec![table_input],
            table_outputs: vec![table_output],
            advice_inputs: vec![advice_input],
            advice_outputs: vec![advice_output],
            s_lookup,
            table_index: None,
            _marker: PhantomData,
        }
    }

    /// Configures a multi-column lookup.
    pub fn configure_multi(
        meta: &mut ConstraintSystem<F>,
        num_inputs: usize,
        num_outputs: usize,
        advice_columns: &[Column<Advice>],
    ) -> PlookupTableConfig<F> {
        assert!(
            advice_columns.len() >= num_inputs + num_outputs,
            "Need at least {} advice columns",
            num_inputs + num_outputs
        );

        let table_inputs: Vec<_> = (0..num_inputs)
            .map(|_| meta.lookup_table_column())
            .collect();
        let table_outputs: Vec<_> = (0..num_outputs)
            .map(|_| meta.lookup_table_column())
            .collect();
        let s_lookup = meta.complex_selector();

        let advice_inputs: Vec<_> = advice_columns[..num_inputs].to_vec();
        let advice_outputs: Vec<_> = advice_columns[num_inputs..num_inputs + num_outputs].to_vec();

        meta.lookup("table_range_check", |meta| {
            let s = meta.query_selector(s_lookup);
            let mut lookup_pairs = Vec::with_capacity(num_inputs + num_outputs);

            for (_i, (table_col, advice_col)) in table_inputs.iter().zip(advice_inputs.iter()).enumerate() {
                let input = meta.query_advice(*advice_col, Rotation::cur());
                lookup_pairs.push((s.clone() * input, *table_col));
            }

            for (table_col, advice_col) in table_outputs.iter().zip(advice_outputs.iter()) {
                let output = meta.query_advice(*advice_col, Rotation::cur());
                lookup_pairs.push((s.clone() * output, *table_col));
            }

            lookup_pairs
        });

        PlookupTableConfig {
            table_inputs,
            table_outputs,
            advice_inputs,
            advice_outputs,
            s_lookup,
            table_index: None,
            _marker: PhantomData,
        }
    }

    /// Loads a table into the circuit.
    pub fn load_table(
        &self,
        layouter: &mut impl Layouter<F>,
        table: &PlookupTable<F>,
    ) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "plookup table",
            |mut table_layouter| {
                // Ensure (0, 0, ...) is the first entry for selector-off case
                let zero_entry: Vec<F> = vec![F::ZERO; self.config.table_inputs.len()];
                let zero_output: Vec<F> = vec![F::ZERO; self.config.table_outputs.len()];

                // Assign zero entry first
                for (col_idx, table_col) in self.config.table_inputs.iter().enumerate() {
                    table_layouter.assign_cell(
                        || "zero_input",
                        *table_col,
                        0,
                        || Value::known(zero_entry.get(col_idx).copied().unwrap_or(F::ZERO)),
                    )?;
                }
                for (col_idx, table_col) in self.config.table_outputs.iter().enumerate() {
                    table_layouter.assign_cell(
                        || "zero_output",
                        *table_col,
                        0,
                        || Value::known(zero_output.get(col_idx).copied().unwrap_or(F::ZERO)),
                    )?;
                }

                // Assign remaining entries
                for (i, (inputs, outputs)) in table.entries().iter().enumerate() {
                    let row = i + 1; // Skip row 0 which has zeros

                    for (col_idx, table_col) in self.config.table_inputs.iter().enumerate() {
                        let val = inputs.get(col_idx).copied().unwrap_or(F::ZERO);
                        table_layouter.assign_cell(
                            || format!("input_{}_{}", row, col_idx),
                            *table_col,
                            row,
                            || Value::known(val),
                        )?;
                    }

                    for (col_idx, table_col) in self.config.table_outputs.iter().enumerate() {
                        let val = outputs.get(col_idx).copied().unwrap_or(F::ZERO);
                        table_layouter.assign_cell(
                            || format!("output_{}_{}", row, col_idx),
                            *table_col,
                            row,
                            || Value::known(val),
                        )?;
                    }
                }

                Ok(())
            },
        )
    }

    /// Loads a table with padding to fill the circuit.
    pub fn load_table_padded(
        &self,
        layouter: &mut impl Layouter<F>,
        table: &PlookupTable<F>,
        min_rows: usize,
    ) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "plookup table (padded)",
            |mut table_layouter| {
                let total_rows = table.len().max(min_rows) + 1; // +1 for zero entry

                for row in 0..total_rows {
                    let (inputs, outputs) = if row == 0 {
                        // Zero entry for selector-off case
                        (
                            vec![F::ZERO; self.config.table_inputs.len()],
                            vec![F::ZERO; self.config.table_outputs.len()],
                        )
                    } else if row <= table.len() {
                        let (inp, out) = &table.entries()[row - 1];
                        (inp.clone(), out.clone())
                    } else {
                        // Padding with zeros
                        (
                            vec![F::ZERO; self.config.table_inputs.len()],
                            vec![F::ZERO; self.config.table_outputs.len()],
                        )
                    };

                    for (col_idx, table_col) in self.config.table_inputs.iter().enumerate() {
                        let val = inputs.get(col_idx).copied().unwrap_or(F::ZERO);
                        table_layouter.assign_cell(
                            || format!("input_{}_{}", row, col_idx),
                            *table_col,
                            row,
                            || Value::known(val),
                        )?;
                    }

                    for (col_idx, table_col) in self.config.table_outputs.iter().enumerate() {
                        let val = outputs.get(col_idx).copied().unwrap_or(F::ZERO);
                        table_layouter.assign_cell(
                            || format!("output_{}_{}", row, col_idx),
                            *table_col,
                            row,
                            || Value::known(val),
                        )?;
                    }
                }

                Ok(())
            },
        )
    }

    /// Performs a single lookup.
    pub fn lookup_single(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_lookup.enable(region, row)?;

        region.assign_advice(
            || format!("lookup_input_{}", row),
            self.config.advice_inputs[0],
            row,
            || input,
        )?;

        region.assign_advice(
            || format!("lookup_output_{}", row),
            self.config.advice_outputs[0],
            row,
            || output,
        )?;

        Ok(())
    }

    /// Performs a multi-column lookup.
    pub fn lookup_multi(
        &self,
        region: &mut Region<'_, F>,
        row: usize,
        inputs: &[Value<F>],
        outputs: &[Value<F>],
    ) -> Result<(), ErrorFront> {
        self.config.s_lookup.enable(region, row)?;

        for (i, (col, val)) in self.config.advice_inputs.iter().zip(inputs.iter()).enumerate() {
            region.assign_advice(
                || format!("lookup_input_{}_{}", row, i),
                *col,
                row,
                || *val,
            )?;
        }

        for (i, (col, val)) in self.config.advice_outputs.iter().zip(outputs.iter()).enumerate() {
            region.assign_advice(
                || format!("lookup_output_{}_{}", row, i),
                *col,
                row,
                || *val,
            )?;
        }

        Ok(())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &PlookupTableConfig<F> {
        &self.config
    }
}

/// Multi-column lookup support for complex activation functions.
#[derive(Clone, Debug)]
pub struct MultiColumnLookup<F: PrimeField, const N_IN: usize, const N_OUT: usize> {
    table: PlookupTable<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const N_IN: usize, const N_OUT: usize> MultiColumnLookup<F, N_IN, N_OUT> {
    /// Creates a new multi-column lookup table.
    pub fn new(scale: u64) -> Self {
        Self {
            table: PlookupTable::new(scale),
            _marker: PhantomData,
        }
    }

    /// Adds an entry to the multi-column table.
    pub fn add_entry(&mut self, inputs: [F; N_IN], outputs: [F; N_OUT]) {
        self.table.add_entry(inputs.to_vec(), outputs.to_vec());
    }

    /// Looks up values in the table.
    pub fn lookup(&self, inputs: &[F; N_IN]) -> Option<[F; N_OUT]> {
        self.table.lookup(inputs.as_slice()).and_then(|v| {
            if v.len() == N_OUT {
                let mut arr = [F::ZERO; N_OUT];
                arr.copy_from_slice(&v);
                Some(arr)
            } else {
                None
            }
        })
    }

    /// Returns the underlying table.
    pub fn inner(&self) -> &PlookupTable<F> {
        &self.table
    }
}

/// Batch lookup optimizer for amortizing lookup costs.
#[derive(Clone, Debug)]
pub struct BatchLookupOptimizer<F: PrimeField> {
    /// Pending lookups to be batched.
    pending: Vec<(Vec<F>, Vec<F>)>,
    /// Maximum batch size before auto-flush.
    max_batch_size: usize,
    /// Statistics.
    stats: LookupStats,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> BatchLookupOptimizer<F> {
    /// Creates a new batch optimizer.
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            pending: Vec::with_capacity(max_batch_size),
            max_batch_size,
            stats: LookupStats::default(),
            _marker: PhantomData,
        }
    }

    /// Adds a lookup to the batch.
    pub fn add(&mut self, inputs: Vec<F>, outputs: Vec<F>) {
        self.pending.push((inputs, outputs));
        self.stats.total_lookups += 1;
    }

    /// Returns whether the batch should be flushed.
    pub fn should_flush(&self) -> bool {
        self.pending.len() >= self.max_batch_size
    }

    /// Flushes the pending batch.
    pub fn flush(&mut self) -> Vec<(Vec<F>, Vec<F>)> {
        std::mem::take(&mut self.pending)
    }

    /// Returns the pending count.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Builder for constructing lookup tables.
#[derive(Clone, Debug)]
pub struct LookupTableBuilder<F: PrimeField> {
    entries: Vec<(Vec<F>, Vec<F>)>,
    scale: u64,
    error_bound: Option<F>,
    _name: String,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> LookupTableBuilder<F> {
    /// Creates a new builder.
    pub fn new(name: &str, scale: u64) -> Self {
        Self {
            entries: Vec::new(),
            scale,
            error_bound: None,
            _name: name.to_string(),
            _marker: PhantomData,
        }
    }

    /// Adds a single entry.
    pub fn entry(mut self, input: F, output: F) -> Self {
        self.entries.push((vec![input], vec![output]));
        self
    }

    /// Adds a multi-column entry.
    pub fn multi_entry(mut self, inputs: Vec<F>, outputs: Vec<F>) -> Self {
        self.entries.push((inputs, outputs));
        self
    }

    /// Adds entries from a range and function.
    pub fn from_range<G>(mut self, range: std::ops::Range<i64>, f: G) -> Self
    where
        G: Fn(i64) -> f64,
    {
        let scale_f = self.scale as f64;
        for x in range {
            let y = f(x);
            let quantized = (y * scale_f).round() as i64;

            let input = if x >= 0 {
                F::from(x as u64)
            } else {
                F::ZERO - F::from((-x) as u64)
            };

            let output = if quantized >= 0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            self.entries.push((vec![input], vec![output]));
        }
        self
    }

    /// Sets the error bound.
    pub fn with_error_bound(mut self, bound: F) -> Self {
        self.error_bound = Some(bound);
        self
    }

    /// Builds the lookup table.
    pub fn build(self) -> PlookupTable<F> {
        let mut table = PlookupTable::new(self.scale);
        for (inputs, outputs) in self.entries {
            table.add_entry(inputs, outputs);
        }
        if let Some(bound) = self.error_bound {
            table.set_error_bound(bound);
        }
        table.stats.table_size = table.len();
        table
    }
}

/// Precomputed table for constant lookup functions.
///
/// This stores the table in a format optimized for circuit loading.
#[derive(Clone, Debug)]
pub struct PrecomputedTable<F: PrimeField> {
    /// Raw entries as field elements.
    entries: Vec<(F, F)>,
    /// Function name for debugging.
    _name: String,
    /// Error bound.
    error_bound: F,
    /// Scale factor.
    scale: u64,
}

impl<F: PrimeField> PrecomputedTable<F> {
    /// Creates a precomputed ReLU table.
    pub fn relu(range: usize) -> Self {
        let half = range / 2;
        let mut entries = Vec::with_capacity(range);

        // Zero entry
        entries.push((F::ZERO, F::ZERO));

        // Positive values: x -> x
        for x in 1..half {
            entries.push((F::from(x as u64), F::from(x as u64)));
        }

        // Negative values: -x -> 0
        for x in 1..half {
            let neg_x = F::ZERO - F::from(x as u64);
            entries.push((neg_x, F::ZERO));
        }

        Self {
            entries,
            _name: format!("relu_{}", range),
            error_bound: F::ZERO, // ReLU is exact
            scale: 1,
        }
    }

    /// Creates a precomputed GELU table.
    pub fn gelu(range: usize, scale: u64) -> Self {
        let half = range / 2;
        let scale_f = scale as f64;
        let mut entries = Vec::with_capacity(range);
        let mut max_error = 0.0f64;

        // GELU: x * Φ(x) where Φ is the standard normal CDF
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();

        for i in 0..range {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((range - i) as f64) / scale_f
            };

            let gelu = 0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh());
            let quantized = (gelu * scale_f).round();
            let error = (gelu - quantized / scale_f).abs();
            max_error = max_error.max(error);

            let input_field = if i < half {
                F::from(i as u64)
            } else {
                F::ZERO - F::from((range - i) as u64)
            };

            let output_field = if quantized >= 0.0 {
                F::from(quantized as u64)
            } else {
                F::ZERO - F::from((-quantized) as u64)
            };

            entries.push((input_field, output_field));
        }

        Self {
            entries,
            _name: format!("gelu_{}_{}", range, scale),
            error_bound: F::from((max_error * scale_f).ceil() as u64 + 1),
            scale,
        }
    }

    /// Creates a precomputed sigmoid table.
    pub fn sigmoid(range: usize, scale: u64) -> Self {
        let half = range / 2;
        let scale_f = scale as f64;
        let mut entries = Vec::with_capacity(range);
        let mut max_error = 0.0f64;

        for i in 0..range {
            let x = if i < half {
                i as f64 / scale_f
            } else {
                -((range - i) as f64) / scale_f
            };

            let sigmoid = 1.0 / (1.0 + (-x).exp());
            let quantized = (sigmoid * scale_f).round();
            let error = (sigmoid - quantized / scale_f).abs();
            max_error = max_error.max(error);

            let input_field = if i < half {
                F::from(i as u64)
            } else {
                F::ZERO - F::from((range - i) as u64)
            };

            entries.push((input_field, F::from(quantized as u64)));
        }

        Self {
            entries,
            _name: format!("sigmoid_{}_{}", range, scale),
            error_bound: F::from((max_error * scale_f).ceil() as u64 + 1),
            scale,
        }
    }

    /// Returns the entries.
    pub fn entries(&self) -> &[(F, F)] {
        &self.entries
    }

    /// Returns the error bound.
    pub fn error_bound(&self) -> F {
        self.error_bound
    }

    /// Converts to a PlookupTable.
    pub fn to_plookup_table(&self) -> PlookupTable<F> {
        let mut table = PlookupTable::new(self.scale);
        for (input, output) in &self.entries {
            table.add_entry(vec![*input], vec![*output]);
        }
        table.set_error_bound(self.error_bound);
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::Fr;
    use halo2curves::ff::Field;

    #[test]
    fn test_plookup_table_creation() {
        let table = PlookupTable::<Fr>::from_function(
            -128..128,
            |x| if x >= 0 { x as f64 } else { 0.0 },
            1,
        );

        assert_eq!(table.len(), 256);

        // Test lookup
        let result = table.lookup(&[Fr::from(5)]);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], Fr::from(5));

        // Test negative (should give 0)
        let neg_5 = Fr::ZERO - Fr::from(5);
        let result = table.lookup(&[neg_5]);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], Fr::ZERO);
    }

    #[test]
    fn test_lookup_table_builder() {
        let table = LookupTableBuilder::<Fr>::new("test", 64)
            .entry(Fr::from(1), Fr::from(2))
            .entry(Fr::from(2), Fr::from(4))
            .entry(Fr::from(3), Fr::from(6))
            .build();

        assert_eq!(table.len(), 3);
        assert_eq!(table.lookup(&[Fr::from(2)]).unwrap()[0], Fr::from(4));
    }

    #[test]
    fn test_precomputed_relu() {
        let table = PrecomputedTable::<Fr>::relu(256);
        assert_eq!(table.entries.len(), 255); // 1 zero + 127 positive + 127 negative

        // Positive should map to itself
        let pos_5 = Fr::from(5);
        let found = table.entries.iter().find(|(inp, _)| *inp == pos_5);
        assert!(found.is_some());
        assert_eq!(found.unwrap().1, pos_5);

        // Negative should map to zero
        let neg_5 = Fr::ZERO - Fr::from(5);
        let found = table.entries.iter().find(|(inp, _)| *inp == neg_5);
        assert!(found.is_some());
        assert_eq!(found.unwrap().1, Fr::ZERO);
    }

    #[test]
    fn test_precomputed_gelu() {
        let table = PrecomputedTable::<Fr>::gelu(256, 64);
        assert_eq!(table.entries.len(), 256);

        // GELU(0) should be approximately 0
        let zero_entry = table.entries.iter().find(|(inp, _)| *inp == Fr::ZERO);
        assert!(zero_entry.is_some());
        // The output should be small (close to 0)
    }

    #[test]
    fn test_batch_optimizer() {
        let mut optimizer = BatchLookupOptimizer::<Fr>::new(10);

        for i in 0..5 {
            optimizer.add(vec![Fr::from(i)], vec![Fr::from(i * 2)]);
        }

        assert_eq!(optimizer.pending_count(), 5);
        assert!(!optimizer.should_flush());

        for i in 5..10 {
            optimizer.add(vec![Fr::from(i)], vec![Fr::from(i * 2)]);
        }

        assert!(optimizer.should_flush());

        let batch = optimizer.flush();
        assert_eq!(batch.len(), 10);
        assert_eq!(optimizer.pending_count(), 0);
    }

    #[test]
    fn test_multi_column_lookup() {
        let mut lookup = MultiColumnLookup::<Fr, 2, 1>::new(1);

        lookup.add_entry([Fr::from(1), Fr::from(2)], [Fr::from(3)]);
        lookup.add_entry([Fr::from(2), Fr::from(3)], [Fr::from(5)]);

        let result = lookup.lookup(&[Fr::from(1), Fr::from(2)]);
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], Fr::from(3));
    }
}
