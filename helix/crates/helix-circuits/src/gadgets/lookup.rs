//! Lookup table gadgets for efficient non-linear operations.
//!
//! Provides lookup-based verification for operations that are expensive to
//! express as polynomial constraints (activation functions, exp, etc.).
//!
//! # Architecture
//!
//! - `LookupTableChip`: Generic two-column (input→output) lookup
//! - `ReLUTableChip`: Piecewise-linear ReLU for quantized integers
//! - `ExpTableChip`: Scaled integer exp approximation for softmax

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{
        Advice, Column, ConstraintSystem, ErrorFront, Selector, TableColumn,
    },
    poly::Rotation,
};
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Generic Lookup Table
// ---------------------------------------------------------------------------

/// Configuration for a generic two-column lookup table.
#[derive(Clone, Debug)]
pub struct LookupTableConfig<F: PrimeField> {
    /// Table column for inputs.
    pub table_input: TableColumn,
    /// Table column for outputs.
    pub table_output: TableColumn,
    /// Advice column for the value being looked up.
    pub advice_input: Column<Advice>,
    /// Advice column for the expected output.
    pub advice_output: Column<Advice>,
    /// Selector to enable the lookup.
    pub s_lookup: Selector,
    _marker: PhantomData<F>,
}

/// Generic lookup table chip.
///
/// Loads an arbitrary set of `(input, output)` pairs and constrains that
/// any enabled row's `(advice_input, advice_output)` appears in the table.
pub struct LookupTableChip<F: PrimeField> {
    config: LookupTableConfig<F>,
}

impl<F: PrimeField> LookupTableChip<F> {
    /// Creates a new chip from an existing configuration.
    pub fn new(config: LookupTableConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the lookup table in the constraint system.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> LookupTableConfig<F> {
        let table_input = meta.lookup_table_column();
        let table_output = meta.lookup_table_column();
        let s_lookup = meta.complex_selector();

        meta.lookup("gadget_lookup", |meta| {
            let s = meta.query_selector(s_lookup);
            let input = meta.query_advice(advice_input, Rotation::cur());
            let output = meta.query_advice(advice_output, Rotation::cur());

            // When selector is off, look up (0, 0) which must be in the table.
            vec![
                (s.clone() * input, table_input),
                (s * output, table_output),
            ]
        });

        LookupTableConfig {
            table_input,
            table_output,
            advice_input,
            advice_output,
            s_lookup,
            _marker: PhantomData,
        }
    }

    /// Loads the lookup table with the given `(input, output)` pairs.
    ///
    /// The table **must** include `(0, 0)` as the first entry (for the
    /// selector-off case where expressions evaluate to zero).
    pub fn load(
        &self,
        layouter: &mut impl Layouter<F>,
        entries: &[(F, F)],
    ) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "generic lookup table",
            |mut table| {
                for (i, (inp, out)) in entries.iter().enumerate() {
                    table.assign_cell(
                        || format!("input_{}", i),
                        self.config.table_input,
                        i,
                        || Value::known(*inp),
                    )?;
                    table.assign_cell(
                        || format!("output_{}", i),
                        self.config.table_output,
                        i,
                        || Value::known(*out),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Loads the lookup table, padding with `(0, 0)` to fill the circuit.
    ///
    /// Use this when `entries.len() < 2^k` to avoid lookup failures on
    /// unused rows.
    pub fn load_padded(
        &self,
        layouter: &mut impl Layouter<F>,
        entries: &[(F, F)],
        min_rows: usize,
    ) -> Result<(), ErrorFront> {
        layouter.assign_table(
            || "generic lookup table (padded)",
            |mut table| {
                let total = entries.len().max(min_rows);
                for i in 0..total {
                    let (inp, out) = if i < entries.len() {
                        entries[i]
                    } else {
                        (F::ZERO, F::ZERO)
                    };
                    table.assign_cell(
                        || format!("input_{}", i),
                        self.config.table_input,
                        i,
                        || Value::known(inp),
                    )?;
                    table.assign_cell(
                        || format!("output_{}", i),
                        self.config.table_output,
                        i,
                        || Value::known(out),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Assigns a lookup: constrains that `(input, output)` is in the table.
    pub fn lookup(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.config.s_lookup.enable(region, row)?;

        region.assign_advice(
            || format!("lookup_input_{}", row),
            self.config.advice_input,
            row,
            || input,
        )?;

        region.assign_advice(
            || format!("lookup_output_{}", row),
            self.config.advice_output,
            row,
            || output,
        )?;

        Ok(())
    }

    /// Returns a reference to the config.
    pub fn config(&self) -> &LookupTableConfig<F> {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// ReLU Lookup Table
// ---------------------------------------------------------------------------

/// Configuration for the ReLU lookup table.
#[derive(Clone, Debug)]
pub struct ReLUTableConfig<F: PrimeField> {
    pub inner: LookupTableConfig<F>,
    _marker: PhantomData<F>,
}

/// ReLU lookup table chip.
///
/// For quantized integer inputs in `[-HALF_RANGE, HALF_RANGE)`, maps:
///   - `x >= 0  →  x`
///   - `x < 0   →  0`
///
/// Internally the "negative" values are represented as field elements
/// `p - |x|` (two's-complement-like in the prime field).
pub struct ReLUTableChip<F: PrimeField, const RANGE: usize> {
    inner: LookupTableChip<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> ReLUTableChip<F, RANGE> {
    /// Creates a new ReLU chip from the inner lookup config.
    pub fn new(config: LookupTableConfig<F>) -> Self {
        Self {
            inner: LookupTableChip::new(config),
            _marker: PhantomData,
        }
    }

    /// Configures the ReLU lookup.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> LookupTableConfig<F> {
        LookupTableChip::configure(meta, advice_input, advice_output)
    }

    /// Loads the ReLU table.
    ///
    /// Entries cover `[0, HALF_RANGE)` (positive) and their negative
    /// counterparts `(p - x, 0)` for `x in [1, HALF_RANGE)`.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        let half = RANGE / 2;
        let entries = relu_entries::<F>(half);
        self.inner.load(layouter, &entries)
    }

    /// Loads the ReLU table with padding for circuits of size `2^k`.
    pub fn load_padded(
        &self,
        layouter: &mut impl Layouter<F>,
        min_rows: usize,
    ) -> Result<(), ErrorFront> {
        let half = RANGE / 2;
        let entries = relu_entries::<F>(half);
        self.inner.load_padded(layouter, &entries, min_rows)
    }

    /// Performs a ReLU lookup.
    pub fn relu(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.inner.lookup(region, row, input, output)
    }
}

// ---------------------------------------------------------------------------
// Exp Lookup Table (scaled integer approximation)
// ---------------------------------------------------------------------------

/// Configuration for the exp approximation table.
#[derive(Clone, Debug)]
pub struct ExpTableConfig<F: PrimeField> {
    pub inner: LookupTableConfig<F>,
    _marker: PhantomData<F>,
}

/// Exp lookup table chip.
///
/// Maps `x` (integer) to `round(SCALE * exp(x / SCALE))` for `x in [0, RANGE)`.
/// Used for numerically-stable softmax computation on quantized values.
pub struct ExpTableChip<F: PrimeField, const RANGE: usize, const SCALE: u64> {
    inner: LookupTableChip<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize, const SCALE: u64> ExpTableChip<F, RANGE, SCALE> {
    /// Creates a new exp table chip.
    pub fn new(config: LookupTableConfig<F>) -> Self {
        Self {
            inner: LookupTableChip::new(config),
            _marker: PhantomData,
        }
    }

    /// Configures the exp lookup.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advice_input: Column<Advice>,
        advice_output: Column<Advice>,
    ) -> LookupTableConfig<F> {
        LookupTableChip::configure(meta, advice_input, advice_output)
    }

    /// Loads the exp approximation table.
    ///
    /// Entry `i` maps `i → round(SCALE * exp(i / SCALE))`.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), ErrorFront> {
        let entries = exp_entries::<F>(RANGE, SCALE);
        self.inner.load(layouter, &entries)
    }

    /// Loads the exp table with padding for circuits of size `2^k`.
    pub fn load_padded(
        &self,
        layouter: &mut impl Layouter<F>,
        min_rows: usize,
    ) -> Result<(), ErrorFront> {
        let entries = exp_entries::<F>(RANGE, SCALE);
        self.inner.load_padded(layouter, &entries, min_rows)
    }

    /// Performs an exp lookup.
    pub fn exp_lookup(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, F>,
        row: usize,
        input: Value<F>,
        output: Value<F>,
    ) -> Result<(), ErrorFront> {
        self.inner.lookup(region, row, input, output)
    }
}

// ---------------------------------------------------------------------------
// Helper: generate table entries for external use
// ---------------------------------------------------------------------------

/// Generates ReLU table entries for a given half-range.
pub fn relu_entries<F: PrimeField>(half_range: usize) -> Vec<(F, F)> {
    let mut entries = Vec::with_capacity(half_range * 2);
    entries.push((F::ZERO, F::ZERO));
    for x in 1..half_range {
        entries.push((F::from(x as u64), F::from(x as u64)));
    }
    for x in 1..half_range {
        entries.push((F::ZERO - F::from(x as u64), F::ZERO));
    }
    entries
}

/// Generates exp approximation entries.
pub fn exp_entries<F: PrimeField>(range: usize, scale: u64) -> Vec<(F, F)> {
    let scale_f = scale as f64;
    (0..range)
        .map(|x| {
            let input = F::from(x as u64);
            let exp_val = ((x as f64) / scale_f).exp() * scale_f;
            let output = F::from(exp_val.round().max(0.0) as u64);
            (input, output)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        arithmetic::Field,
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::Circuit,
    };
    use halo2curves::bn256::Fr;

    // ---- Generic Lookup Test Circuit ----

    #[derive(Clone)]
    struct GenericLookupCircuit {
        entries: Vec<(Fr, Fr)>,
        input: Fr,
        output: Fr,
    }

    impl Default for GenericLookupCircuit {
        fn default() -> Self {
            Self {
                entries: vec![(Fr::ZERO, Fr::ZERO)],
                input: Fr::ZERO,
                output: Fr::ZERO,
            }
        }
    }

    impl Circuit<Fr> for GenericLookupCircuit {
        type Config = LookupTableConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice_in = meta.advice_column();
            let advice_out = meta.advice_column();
            meta.enable_equality(advice_in);
            meta.enable_equality(advice_out);
            LookupTableChip::configure(meta, advice_in, advice_out)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = LookupTableChip::new(config.clone());

            // Pad table to fill circuit rows (avoids lookup failures on unused rows).
            chip.load_padded(&mut layouter, &self.entries, 1024)?;

            layouter.assign_region(
                || "lookup",
                |mut region| {
                    chip.lookup(
                        &mut region,
                        0,
                        Value::known(self.input),
                        Value::known(self.output),
                    )
                },
            )
        }
    }

    #[test]
    fn test_generic_lookup_valid() {
        // Table: (0,0), (1,1), (2,4), (3,9) -- squares
        let entries = vec![
            (Fr::from(0), Fr::from(0)),
            (Fr::from(1), Fr::from(1)),
            (Fr::from(2), Fr::from(4)),
            (Fr::from(3), Fr::from(9)),
        ];

        let circuit = GenericLookupCircuit {
            entries: entries.clone(),
            input: Fr::from(2),
            output: Fr::from(4),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_generic_lookup_invalid() {
        let entries = vec![
            (Fr::from(0), Fr::from(0)),
            (Fr::from(1), Fr::from(1)),
            (Fr::from(2), Fr::from(4)),
            (Fr::from(3), Fr::from(9)),
        ];

        // (2, 5) is NOT in the table
        let circuit = GenericLookupCircuit {
            entries,
            input: Fr::from(2),
            output: Fr::from(5),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    // ---- ReLU Test Circuit ----

    #[derive(Clone)]
    struct ReLUCircuit {
        input: Fr,
        output: Fr,
    }

    impl Default for ReLUCircuit {
        fn default() -> Self {
            Self {
                input: Fr::ZERO,
                output: Fr::ZERO,
            }
        }
    }

    impl Circuit<Fr> for ReLUCircuit {
        type Config = LookupTableConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice_in = meta.advice_column();
            let advice_out = meta.advice_column();
            meta.enable_equality(advice_in);
            meta.enable_equality(advice_out);
            ReLUTableChip::<Fr, 256>::configure(meta, advice_in, advice_out)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = ReLUTableChip::<Fr, 256>::new(config.clone());
            chip.load_padded(&mut layouter, 1024)?;

            layouter.assign_region(
                || "relu",
                |mut region| {
                    chip.relu(
                        &mut region,
                        0,
                        Value::known(self.input),
                        Value::known(self.output),
                    )
                },
            )
        }
    }

    #[test]
    fn test_relu_positive() {
        // relu(5) = 5
        let circuit = ReLUCircuit {
            input: Fr::from(5),
            output: Fr::from(5),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_relu_zero() {
        // relu(0) = 0
        let circuit = ReLUCircuit {
            input: Fr::ZERO,
            output: Fr::ZERO,
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_relu_negative() {
        // relu(-3) = 0, where -3 in field = p - 3
        let neg3 = Fr::ZERO - Fr::from(3);
        let circuit = ReLUCircuit {
            input: neg3,
            output: Fr::ZERO,
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    // ---- Exp Test Circuit ----

    #[derive(Clone)]
    struct ExpCircuit {
        input: Fr,
        output: Fr,
    }

    impl Default for ExpCircuit {
        fn default() -> Self {
            Self {
                input: Fr::ZERO,
                output: Fr::ZERO,
            }
        }
    }

    impl Circuit<Fr> for ExpCircuit {
        type Config = LookupTableConfig<Fr>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let advice_in = meta.advice_column();
            let advice_out = meta.advice_column();
            meta.enable_equality(advice_in);
            meta.enable_equality(advice_out);
            ExpTableChip::<Fr, 128, 64>::configure(meta, advice_in, advice_out)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = ExpTableChip::<Fr, 128, 64>::new(config.clone());
            chip.load_padded(&mut layouter, 1024)?;

            layouter.assign_region(
                || "exp",
                |mut region| {
                    chip.exp_lookup(
                        &mut region,
                        0,
                        Value::known(self.input),
                        Value::known(self.output),
                    )
                },
            )
        }
    }

    #[test]
    fn test_exp_zero() {
        // exp(0) = 1.0, scaled by 64 → 64
        let circuit = ExpCircuit {
            input: Fr::from(0),
            output: Fr::from(64), // SCALE * exp(0/SCALE) = 64 * 1 = 64
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_exp_one_scale() {
        // exp(64/64) = exp(1) ≈ 2.718, scaled by 64 → round(174) = 174
        let expected = (1.0_f64.exp() * 64.0).round() as u64;
        let circuit = ExpCircuit {
            input: Fr::from(64),
            output: Fr::from(expected),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn test_exp_invalid() {
        // (0, 999) is not in the table
        let circuit = ExpCircuit {
            input: Fr::from(0),
            output: Fr::from(999),
        };

        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    // ---- Entry generator tests ----

    #[test]
    fn test_relu_entries_generation() {
        let entries = relu_entries::<Fr>(16);
        // Should have 1 + 15 + 15 = 31 entries
        assert_eq!(entries.len(), 31);

        // (0, 0) must be present
        assert!(entries.contains(&(Fr::ZERO, Fr::ZERO)));

        // (5, 5) must be present (positive)
        assert!(entries.contains(&(Fr::from(5), Fr::from(5))));

        // (p-5, 0) must be present (negative → 0)
        let neg5 = Fr::ZERO - Fr::from(5);
        assert!(entries.contains(&(neg5, Fr::ZERO)));
    }

    #[test]
    fn test_exp_entries_generation() {
        let entries = exp_entries::<Fr>(32, 64);
        assert_eq!(entries.len(), 32);

        // exp(0) = SCALE
        assert_eq!(entries[0], (Fr::from(0), Fr::from(64)));
    }
}
